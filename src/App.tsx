import { useState, useEffect, useRef } from 'react'
import { AutoTokenizer, env } from '@huggingface/transformers'
import { Search, Zap, Loader2, FileText, BookOpen, Scale, AlertTriangle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

// Configuration for transformers.js
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = '/';

// Helper to convert Float16 (2 bytes) to Float32
function f16ToF32(h: number) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7C00) >> 10;
  const f = h & 0x03FF;
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  } else if (e === 0x1F) {
    return f ? NaN : (s ? -Infinity : Infinity);
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function parseSafeTensors(buffer: ArrayBuffer) {
  const view = new DataView(buffer);
  const headerSize = Number(view.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(new Uint8Array(buffer, 8, headerSize));
  const header = JSON.parse(headerJson);
  const offset = 8 + headerSize;

  const tensors: Record<string, { data: any; shape: number[]; dtype: string }> = {};

  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const { dtype, shape, data_offsets } = info as any;
    const start = offset + data_offsets[0];
    const end = offset + data_offsets[1];
    const length = end - start;
    
    let data;
    if (dtype === 'F32') {
      data = new Float32Array(buffer, start, length / 4);
    } else if (dtype === 'F16') {
      const f16Data = new Uint16Array(buffer, start, length / 2);
      data = new Float32Array(f16Data.length);
      for (let i = 0; i < f16Data.length; i++) {
        data[i] = f16ToF32(f16Data[i]);
      }
    } else if (dtype === 'F64') {
      data = new Float64Array(buffer, start, length / 8);
    } else {
      data = new Uint8Array(buffer, start, length);
    }

    tensors[name] = { data, shape, dtype };
  }

  return tensors;
}

interface Article {
  title: string;
  chapter: string;
  section: string;
  article: string;
  page_number: number;
  article_content: string;
  embeddingMatrix?: Float32Array[];
}

function App() {
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle')
  const [progress, setProgress] = useState(0)
  const [articles, setArticles] = useState<Article[]>([])
  const [similarities, setSimilarities] = useState<any[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [stats, setStats] = useState<any>(null)
  
  const tensorsRef = useRef<any>(null)
  const tokenizerRef = useRef<any>(null)

  useEffect(() => {
    const initApp = async () => {
      try {
        setStatus('loading')
        setProgress(5)

        const tokenizer = await AutoTokenizer.from_pretrained('custom_snowflake_onnx')
        tokenizerRef.current = tokenizer
        setProgress(15)

        const weightsResponse = await fetch('/custom_snowflake_onnx/model.safetensors')
        const weightsBuffer = await weightsResponse.arrayBuffer()
        tensorsRef.current = parseSafeTensors(weightsBuffer)
        setProgress(30)

        const articlesResponse = await fetch('/moldovan_constitution_articles_with_content.json')
        const articlesData: Article[] = await articlesResponse.json()
        
        console.log('Indexing articles...')
        const total = articlesData.length
        for (let i = 0; i < total; i++) {
          const article = articlesData[i]
          const fullText = [
            article.title,
            article.chapter,
            article.section,
            article.article,
            article.article_content
          ].filter(Boolean).join(' ')
          
          article.embeddingMatrix = await getTokenVectors(fullText)
          
          if (i % 10 === 0) {
            setProgress(30 + (i / total) * 70)
          }
        }
        
        setArticles(articlesData)
        setStatus('ready')
        setProgress(100)
      } catch (error: any) {
        console.error('Initialization failed:', error)
        setStatus('error')
        setErrorMessage(error.message)
      }
    }

    initApp()
  }, [])

  const getTokenVectors = async (text: string) => {
    if (!tensorsRef.current || !tokenizerRef.current) return [];
    
    const encoded = await tokenizerRef.current([text], { padding: false, truncation: true })
    const inputIds = encoded.input_ids.tolist()[0]
    
    const { embeddings } = tensorsRef.current;
    const dim = embeddings.shape[1];
    const vectors: Float32Array[] = [];
    
    for (let id of inputIds) {
      id = Number(id);
      if (id >= embeddings.shape[0]) continue;
      
      const vec = new Float32Array(dim);
      const offset = id * dim;
      
      let magnitude = 0;
      for (let d = 0; d < dim; d++) {
        const val = embeddings.data[offset + d];
        vec[d] = val;
        magnitude += val * val;
      }
      magnitude = Math.sqrt(magnitude);
      for (let d = 0; d < dim; d++) vec[d] /= (magnitude || 1e-12);
      
      vectors.push(vec);
    }
    
    return vectors;
  }

  const calculateMaxSim = (queryVectors: Float32Array[], docVectors: Float32Array[]) => {
    if (queryVectors.length === 0 || docVectors.length === 0) return 0;
    let totalMaxSim = 0;
    for (const qVec of queryVectors) {
      let maxDot = -Infinity;
      for (const dVec of docVectors) {
        let dot = 0;
        for (let i = 0; i < qVec.length; i++) {
          dot += qVec[i] * dVec[i];
        }
        if (dot > maxDot) maxDot = dot;
      }
      totalMaxSim += maxDot;
    }
    return totalMaxSim / queryVectors.length;
  }

  const handleSearch = async () => {
    if (!query || status !== 'ready') return
    
    setIsProcessing(true)
    try {
      const startTime = performance.now()
      const queryVectors = await getTokenVectors(query);
      
      const scores = articles.map((article) => {
        const score = calculateMaxSim(queryVectors, article.embeddingMatrix || []);
        return { ...article, score };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)

      const endTime = performance.now()
      setSimilarities(scores);
      setStats({
        time: (endTime - startTime).toFixed(1),
        tokens: queryVectors.length
      })
    } catch (error: any) {
      console.error('Search failed:', error)
      setErrorMessage(error.message)
    } finally {
      setIsProcessing(false)
    }
  }

  const openPdf = (page: number) => {
    window.open(`/Constitutia_RM_RO.pdf#page=${page}`, '_blank');
  }

  return (
    <>
      <div className="bg-gradient" />
      
      <div className="container" style={{ paddingBottom: '10rem' }}>
        <header className="header" style={{ marginBottom: '3rem' }}>
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', justifyContent: 'center' }}
          >
            <div className="glass-card" style={{ padding: '1rem', borderRadius: '50%', background: 'rgba(93, 93, 255, 0.15)', border: '1px solid rgba(93, 93, 255, 0.3)' }}>
              <Scale size={48} className="glow-text" style={{ color: 'var(--accent)' }} />
            </div>
            <div style={{ textAlign: 'left' }}>
              <h1 className="title" style={{ margin: 0, fontSize: '2.5rem' }}>Constituția RM</h1>
              <p className="subtitle" style={{ margin: 0, opacity: 0.8 }}>
                Sistem Inteligent de Căutare Semantică (Static ColBERT)
              </p>
            </div>
          </motion.div>
        </header>

        <main className="main-content">
          <motion.div 
            className="glass-card result-card"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            style={{ marginBottom: '3rem' }}
          >
            <div className="input-group">
              <label style={{ fontSize: '0.8rem', color: 'var(--fg-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                Căutare în conținutul Constituției
              </label>
              <div className="input-container">
                <Search className="search-icon" size={20} />
                <input 
                  type="text" 
                  className="search-input" 
                  placeholder="Ex: Care sunt drepturile omului? sau Forma de guvernămînt..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                />
              </div>
              <button 
                className="button-primary" 
                onClick={handleSearch}
                disabled={status !== 'ready' || isProcessing || !query}
                style={{ height: '3.5rem', fontSize: '1.1rem' }}
              >
                {isProcessing ? <Loader2 className="animate-spin" size={20} /> : <Zap size={20} />}
                {isProcessing ? 'Se caută...' : 'Caută Semantic'}
              </button>
            </div>

            <div className="status-indicator">
              <div className={`status-dot ${status === 'loading' ? 'loading' : status === 'ready' ? 'active' : status === 'error' ? 'error' : ''}`} 
                   style={{ backgroundColor: status === 'error' ? '#ef4444' : undefined }} />
              <span>
                {status === 'idle' && 'Inițializare...'}
                {status === 'loading' && `Indexare articole (${Math.round(progress)}%)`}
                {status === 'ready' && `Sistem gata: ${articles.length} articole indexate`}
                {status === 'error' && 'Eroare la încărcare'}
              </span>
            </div>
            
            {status === 'loading' && (
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
            )}

            {errorMessage && (
              <div style={{ marginTop: '1rem', padding: '1rem', borderRadius: 'var(--radius-sm)', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', display: 'flex', gap: '0.75rem', alignItems: 'center', color: '#fca5a5', fontSize: '0.85rem' }}>
                <AlertTriangle size={18} />
                <span>{errorMessage}</span>
              </div>
            )}

            {stats && (
              <div style={{ marginTop: '1rem', fontSize: '0.8rem', color: 'var(--fg-muted)', display: 'flex', gap: '1.5rem', justifyContent: 'center' }}>
                <span>Timp execuție: <strong>{stats.time}ms</strong></span>
                <span>Tokeni interogați: <strong>{stats.tokens}</strong></span>
              </div>
            )}
          </motion.div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <AnimatePresence>
              {similarities.length > 0 ? (
                similarities.map((item, i) => (
                  <motion.div
                    key={`${item.article}-${i}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="glass-card result-card"
                    style={{ 
                      padding: '1.5rem', 
                      textAlign: 'left',
                      borderLeft: '4px solid var(--accent)',
                      background: `linear-gradient(135deg, rgba(93, 93, 255, ${(item.score - 0.4) * 0.2}) 0%, rgba(20, 20, 20, 0.7) 100%)`
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.25rem' }}>
                          <BookOpen size={16} color="var(--accent)" />
                          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase' }}>
                            {item.title}
                          </span>
                        </div>
                        <h3 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 700 }}>{item.article}</h3>
                        {item.chapter && (
                          <div style={{ fontSize: '0.85rem', color: 'var(--fg-muted)', marginTop: '0.25rem' }}>
                            {item.chapter} {item.section && `• ${item.section}`}
                          </div>
                        )}
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '0.7rem', color: 'var(--fg-muted)', textTransform: 'uppercase' }}>Scor MaxSim</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent)' }}>
                          {(item.score * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    <p style={{ fontSize: '1rem', color: 'var(--fg-subtle)', lineHeight: 1.6, whiteSpace: 'pre-wrap', margin: '1rem 0' }}>
                      {item.article_content}
                    </p>

                    <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '1.25rem' }}>
                      <button 
                        onClick={() => openPdf(item.page_number)}
                        className="button-primary"
                        style={{ height: '2.5rem', padding: '0 1.5rem', fontSize: '0.85rem' }}
                      >
                        <FileText size={16} />
                        Vezi în PDF (Pagina {item.page_number})
                      </button>
                    </div>
                  </motion.div>
                ))
              ) : status === 'ready' && (
                <div style={{ textAlign: 'center', padding: '5rem', color: 'var(--fg-muted)' }}>
                  <Search size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} />
                  <p>Introdu o întrebare pentru a căuta în Constituție...</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </main>

        <footer style={{ marginTop: '5rem', padding: '2rem 0', borderTop: '1px solid var(--border-subtle)', textAlign: 'center', color: 'var(--fg-muted)', fontSize: '0.85rem' }}>
          Sistem dezvoltat pentru analiza semantică a textelor juridice • Model: Snowflake Model2Vec (Static ColBERT)
        </footer>
      </div>
    </>
  )
}

export default App
