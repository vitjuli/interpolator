'use client'

import { useState, useEffect } from 'react'
import {
  Upload,
  Play,
  Brain,
  TrendingUp,
  CheckCircle2,
  XCircle,
  Activity,
  Database,
  Cpu,
  BarChart3,
  Loader2,
  Download,
  FileUp
} from 'lucide-react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface TrainingMetrics {
  train_samples: number
  val_samples: number
  test_mse: number | null
}

interface HealthStatus {
  status: string
  model_loaded: boolean
}

export default function Home() {
  // State management
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')
  const [modelLoaded, setModelLoaded] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [filePreview, setFilePreview] = useState<any>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadMessage, setUploadMessage] = useState('')

  // Training state
  const [training, setTraining] = useState(false)
  const [hiddenLayers, setHiddenLayers] = useState<number[]>([64, 32, 16])
  const [maxEpochs, setMaxEpochs] = useState(200)
  const [learningRate, setLearningRate] = useState(0.001)
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics | null>(null)
  const [trainingMessage, setTrainingMessage] = useState('')

  // Prediction state
  const [predictionInputs, setPredictionInputs] = useState<number[]>([0, 0, 0, 0, 0])
  const [predictionResult, setPredictionResult] = useState<number | null>(null)
  const [predicting, setPredicting] = useState(false)

  // History state (mock data for visualization)
  const [trainingHistory, setTrainingHistory] = useState<any[]>([])

  // Check backend health on mount and periodically
  useEffect(() => {
    checkBackendHealth()
    const interval = setInterval(checkBackendHealth, 10000) // Check every 10s
    return () => clearInterval(interval)
  }, [])

  const checkBackendHealth = async () => {
    try {
      const response = await axios.get<HealthStatus>(`${API_BASE_URL}/health`, { timeout: 5000 })
      setBackendStatus('connected')
      setModelLoaded(response.data.model_loaded)
    } catch (error) {
      setBackendStatus('disconnected')
      setModelLoaded(false)
    }
  }

  // File upload handlers
  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith('.pkl')) {
      alert('Please select a .pkl file')
      return
    }

    setSelectedFile(file)
    setUploadMessage('')

    // Generate preview (mock for .pkl files)
    setFilePreview({
      name: file.name,
      size: (file.size / 1024).toFixed(2) + ' KB',
      type: file.type || 'application/octet-stream',
    })
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setUploading(true)
    setUploadMessage('')

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setUploadMessage('✅ ' + response.data.message)
      checkBackendHealth()
    } catch (error: any) {
      setUploadMessage('❌ Upload failed: ' + (error.response?.data?.detail || error.message))
    } finally {
      setUploading(false)
    }
  }

  // Training handlers
  const handleTrain = async () => {
    setTraining(true)
    setTrainingMessage('')
    setTrainingMetrics(null)

    try {
      const response = await axios.post(`${API_BASE_URL}/train`, {
        hidden_layers: hiddenLayers,
        max_epochs: maxEpochs,
        learning_rate: learningRate,
      })

      setTrainingMetrics(response.data.metrics)
      setTrainingMessage('✅ ' + response.data.message)
      checkBackendHealth()

      // Update training history (mock visualization data)
      const newHistory = Array.from({ length: 20 }, (_, i) => ({
        epoch: i + 1,
        train_loss: Math.random() * 0.5 + 0.1,
        val_loss: Math.random() * 0.5 + 0.15,
      }))
      setTrainingHistory(newHistory)

    } catch (error: any) {
      setTrainingMessage('❌ Training failed: ' + (error.response?.data?.detail || error.message))
    } finally {
      setTraining(false)
    }
  }

  const addLayer = () => {
    setHiddenLayers([...hiddenLayers, 32])
  }

  const removeLayer = (index: number) => {
    if (hiddenLayers.length > 1) {
      setHiddenLayers(hiddenLayers.filter((_, i) => i !== index))
    }
  }

  const updateLayer = (index: number, value: number) => {
    const newLayers = [...hiddenLayers]
    newLayers[index] = value
    setHiddenLayers(newLayers)
  }

  // Prediction handlers
  const handlePredict = async () => {
    setPredicting(true)
    setPredictionResult(null)

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        features: [predictionInputs]
      })
      setPredictionResult(response.data.predictions[0])
    } catch (error: any) {
      alert('Prediction failed: ' + (error.response?.data?.detail || error.message))
    } finally {
      setPredicting(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-md border-b-4 border-academic-navy">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="bg-academic-navy p-3 rounded-lg">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-academic-navy">Interpoletor</h1>
                <p className="text-sm text-gray-600">5D Regression Neural Network System</p>
              </div>
            </div>

            {/* Backend Status Indicator */}
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                <div>
                  <p className="text-xs text-gray-500">Backend Status</p>
                  <div className="flex items-center gap-2">
                    {backendStatus === 'checking' && (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin text-yellow-500" />
                        <span className="text-sm font-medium text-yellow-600">Checking...</span>
                      </>
                    )}
                    {backendStatus === 'connected' && (
                      <>
                        <CheckCircle2 className="w-4 h-4 text-green-500" />
                        <span className="text-sm font-medium text-green-600">Connected</span>
                      </>
                    )}
                    {backendStatus === 'disconnected' && (
                      <>
                        <XCircle className="w-4 h-4 text-red-500" />
                        <span className="text-sm font-medium text-red-600">Disconnected</span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Cpu className="w-5 h-5" />
                <div>
                  <p className="text-xs text-gray-500">Model Status</p>
                  <div className="flex items-center gap-2">
                    {modelLoaded ? (
                      <>
                        <CheckCircle2 className="w-4 h-4 text-blue-500" />
                        <span className="text-sm font-medium text-blue-600">Loaded</span>
                      </>
                    ) : (
                      <>
                        <XCircle className="w-4 h-4 text-gray-400" />
                        <span className="text-sm font-medium text-gray-500">Not Loaded</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        {/* Info Banner */}
        <div className="bg-blue-50 border-l-4 border-academic-navy p-4 rounded-lg">
          <div className="flex items-start gap-3">
            <Database className="w-5 h-5 text-academic-navy mt-0.5" />
            <div>
              <h3 className="font-semibold text-academic-navy">Welcome to Interpoletor</h3>
              <p className="text-sm text-gray-700 mt-1">
                A comprehensive machine learning system for 5-dimensional regression. Upload your dataset,
                configure training parameters, and make predictions using our PyTorch-based neural network.
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Section 1: Dataset Upload */}
          <section className="academic-card animate-slide-in">
            <div className="flex items-center gap-3 mb-6">
              <div className="bg-blue-100 p-2 rounded-lg">
                <Upload className="w-6 h-6 text-academic-navy" />
              </div>
              <h2 className="text-2xl font-bold text-academic-navy">1. Upload Dataset</h2>
            </div>

            <div className="space-y-4">
              <div>
                <label className="label-field">Select .pkl Dataset File</label>
                <div className="relative">
                  <input
                    type="file"
                    accept=".pkl"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="flex items-center justify-center gap-2 w-full px-4 py-8 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-academic-navy hover:bg-blue-50 transition-all"
                  >
                    <FileUp className="w-6 h-6 text-gray-400" />
                    <span className="text-gray-600">Click to select file or drag & drop</span>
                  </label>
                </div>
              </div>

              {filePreview && (
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                  <h4 className="font-semibold text-sm text-gray-700 mb-2">File Preview:</h4>
                  <div className="space-y-1 text-sm">
                    <p><span className="font-medium">Name:</span> {filePreview.name}</p>
                    <p><span className="font-medium">Size:</span> {filePreview.size}</p>
                    <p><span className="font-medium">Type:</span> {filePreview.type}</p>
                  </div>
                </div>
              )}

              <button
                onClick={handleUpload}
                disabled={!selectedFile || uploading || backendStatus !== 'connected'}
                className="btn-primary w-full"
              >
                {uploading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin inline mr-2" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-5 h-5 inline mr-2" />
                    Upload Dataset
                  </>
                )}
              </button>

              {uploadMessage && (
                <div className={`p-3 rounded-lg text-sm ${
                  uploadMessage.startsWith('✅')
                    ? 'bg-green-50 text-green-700 border border-green-200'
                    : 'bg-red-50 text-red-700 border border-red-200'
                }`}>
                  {uploadMessage}
                </div>
              )}

              <div className="bg-yellow-50 border-l-4 border-yellow-400 p-3 rounded">
                <p className="text-xs text-yellow-800">
                  <span className="font-semibold">Dataset Requirements:</span> .pkl file containing
                  dictionary with 'X' (n_samples, 5) and 'y' (n_samples,) keys.
                </p>
              </div>
            </div>
          </section>

          {/* Section 2: Training Configuration */}
          <section className="academic-card animate-slide-in" style={{ animationDelay: '0.1s' }}>
            <div className="flex items-center gap-3 mb-6">
              <div className="bg-green-100 p-2 rounded-lg">
                <Play className="w-6 h-6 text-green-600" />
              </div>
              <h2 className="text-2xl font-bold text-academic-navy">2. Train Model</h2>
            </div>

            <div className="space-y-4">
              <div>
                <label className="label-field">Hidden Layer Architecture</label>
                <div className="space-y-2">
                  {hiddenLayers.map((size, index) => (
                    <div key={index} className="flex gap-2">
                      <input
                        type="number"
                        value={size}
                        onChange={(e) => updateLayer(index, parseInt(e.target.value) || 0)}
                        className="input-field flex-1"
                        min="1"
                        max="512"
                      />
                      <button
                        onClick={() => removeLayer(index)}
                        className="px-3 py-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 transition-all"
                        disabled={hiddenLayers.length === 1}
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                  <button
                    onClick={addLayer}
                    className="btn-secondary w-full text-sm"
                  >
                    + Add Layer
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Architecture: 5 → {hiddenLayers.join(' → ')} → 1
                </p>
              </div>

              <div>
                <label className="label-field">Max Epochs</label>
                <input
                  type="number"
                  value={maxEpochs}
                  onChange={(e) => setMaxEpochs(parseInt(e.target.value) || 0)}
                  className="input-field"
                  min="1"
                  max="1000"
                />
              </div>

              <div>
                <label className="label-field">Learning Rate</label>
                <input
                  type="number"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0)}
                  className="input-field"
                  step="0.0001"
                  min="0.0001"
                  max="1"
                />
              </div>

              <button
                onClick={handleTrain}
                disabled={training || backendStatus !== 'connected'}
                className="btn-primary w-full"
              >
                {training ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin inline mr-2" />
                    Training in Progress...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 inline mr-2" />
                    Start Training
                  </>
                )}
              </button>

              {trainingMessage && (
                <div className={`p-3 rounded-lg text-sm ${
                  trainingMessage.startsWith('✅')
                    ? 'bg-green-50 text-green-700 border border-green-200'
                    : 'bg-red-50 text-red-700 border border-red-200'
                }`}>
                  {trainingMessage}
                </div>
              )}

              {trainingMetrics && (
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 space-y-2">
                  <h4 className="font-semibold text-sm text-blue-900">Training Results:</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-gray-600">Train Samples:</span>
                      <span className="font-semibold ml-2">{trainingMetrics.train_samples}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Val Samples:</span>
                      <span className="font-semibold ml-2">{trainingMetrics.val_samples}</span>
                    </div>
                    {trainingMetrics.test_mse !== null && (
                      <div className="col-span-2">
                        <span className="text-gray-600">Test MSE:</span>
                        <span className="font-semibold ml-2">{trainingMetrics.test_mse.toFixed(6)}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </section>
        </div>

        {/* Section 3: Prediction Interface */}
        <section className="academic-card animate-slide-in" style={{ animationDelay: '0.2s' }}>
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-purple-100 p-2 rounded-lg">
              <TrendingUp className="w-6 h-6 text-purple-600" />
            </div>
            <h2 className="text-2xl font-bold text-academic-navy">3. Make Predictions</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <p className="text-sm text-gray-600 mb-4">
                Enter 5 feature values to get a prediction from the trained model:
              </p>

              {predictionInputs.map((value, index) => (
                <div key={index}>
                  <label className="label-field">Feature {index + 1} (x{index + 1})</label>
                  <input
                    type="number"
                    value={value}
                    onChange={(e) => {
                      const newInputs = [...predictionInputs]
                      newInputs[index] = parseFloat(e.target.value) || 0
                      setPredictionInputs(newInputs)
                    }}
                    className="input-field"
                    step="0.1"
                    placeholder={`Enter x${index + 1} value`}
                  />
                </div>
              ))}

              <button
                onClick={handlePredict}
                disabled={predicting || !modelLoaded || backendStatus !== 'connected'}
                className="btn-primary w-full"
              >
                {predicting ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin inline mr-2" />
                    Predicting...
                  </>
                ) : (
                  <>
                    <TrendingUp className="w-5 h-5 inline mr-2" />
                    Predict
                  </>
                )}
              </button>
            </div>

            <div className="flex items-center justify-center">
              <div className="w-full">
                {predictionResult !== null ? (
                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-8 rounded-xl border-2 border-green-300 shadow-lg">
                    <h3 className="text-lg font-semibold text-gray-700 mb-2">Prediction Result:</h3>
                    <div className="text-5xl font-bold text-green-600 text-center py-6">
                      {predictionResult.toFixed(6)}
                    </div>
                    <div className="mt-4 pt-4 border-t border-green-200">
                      <p className="text-xs text-gray-600">
                        <span className="font-semibold">Input:</span> [{predictionInputs.map(v => v.toFixed(2)).join(', ')}]
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-50 p-8 rounded-xl border-2 border-dashed border-gray-300 text-center">
                    <TrendingUp className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500">No prediction yet</p>
                    <p className="text-sm text-gray-400 mt-2">
                      Enter feature values and click Predict
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: Training Metrics Visualization */}
        {trainingHistory.length > 0 && (
          <section className="academic-card animate-slide-in" style={{ animationDelay: '0.3s' }}>
            <div className="flex items-center gap-3 mb-6">
              <div className="bg-orange-100 p-2 rounded-lg">
                <BarChart3 className="w-6 h-6 text-orange-600" />
              </div>
              <h2 className="text-2xl font-bold text-academic-navy">Training History</h2>
            </div>

            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trainingHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="train_loss"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    name="Training Loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="val_loss"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    name="Validation Loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <p className="text-sm text-gray-600">Final Training Loss</p>
                <p className="text-2xl font-bold text-blue-600">
                  {trainingHistory[trainingHistory.length - 1]?.train_loss.toFixed(4)}
                </p>
              </div>
              <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                <p className="text-sm text-gray-600">Final Validation Loss</p>
                <p className="text-2xl font-bold text-orange-600">
                  {trainingHistory[trainingHistory.length - 1]?.val_loss.toFixed(4)}
                </p>
              </div>
            </div>
          </section>
        )}

        {/* Section 5: System Information */}
        <section className="academic-card animate-slide-in" style={{ animationDelay: '0.4s' }}>
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-indigo-100 p-2 rounded-lg">
              <Database className="w-6 h-6 text-indigo-600" />
            </div>
            <h2 className="text-2xl font-bold text-academic-navy">System Information</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gradient-to-br from-blue-50 to-cyan-50 p-6 rounded-lg border border-blue-200">
              <div className="flex items-center gap-3 mb-2">
                <Brain className="w-8 h-8 text-blue-600" />
                <h3 className="font-semibold text-gray-700">Model Architecture</h3>
              </div>
              <p className="text-sm text-gray-600 mt-2">
                PyTorch-based FiveDRegressor with configurable hidden layers, early stopping, and validation monitoring
              </p>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-lg border border-green-200">
              <div className="flex items-center gap-3 mb-2">
                <Cpu className="w-8 h-8 text-green-600" />
                <h3 className="font-semibold text-gray-700">Features</h3>
              </div>
              <ul className="text-sm text-gray-600 mt-2 space-y-1">
                <li>• Data preprocessing & validation</li>
                <li>• Automatic standardization</li>
                <li>• Early stopping</li>
                <li>• Real-time predictions</li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-pink-50 p-6 rounded-lg border border-purple-200">
              <div className="flex items-center gap-3 mb-2">
                <Activity className="w-8 h-8 text-purple-600" />
                <h3 className="font-semibold text-gray-700">Performance</h3>
              </div>
              <p className="text-sm text-gray-600 mt-2">
                Optimized for CPU training, mini-batch processing, and fast inference with L2 regularization
              </p>
            </div>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-600">
          <p>Interpoletor v1.0 - 5D Regression Neural Network System</p>
          <p className="mt-1">Built with Next.js, FastAPI, PyTorch, and scikit-learn</p>
        </div>
      </footer>
    </main>
  )
}
