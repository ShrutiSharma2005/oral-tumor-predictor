import React, { useEffect, useRef, useState, useCallback } from 'react'

const TumorEvolutionVisualization = ({ csvData, imagePaths = [] }) => {
  const canvasRef = useRef(null)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const playIntervalRef = useRef(null)

  // Parse CSV data if it's a string, otherwise use as-is
  const data = React.useMemo(() => {
    if (!csvData) return []
    if (typeof csvData === 'string') {
      try {
        const lines = csvData.split('\n').filter(line => line.trim())
        if (lines.length < 2) return []
        
        // Parse CSV with proper handling of quoted values
        const parseCSVLine = (line) => {
          const result = []
          let current = ''
          let inQuotes = false
          
          for (let i = 0; i < line.length; i++) {
            const char = line[i]
            if (char === '"') {
              inQuotes = !inQuotes
            } else if (char === ',' && !inQuotes) {
              result.push(current.trim())
              current = ''
            } else {
              current += char
            }
          }
          result.push(current.trim())
          return result
        }
        
        const headers = parseCSVLine(lines[0]).map(h => h.replace(/^"|"$/g, '').trim())
        return lines.slice(1).map(line => {
          const values = parseCSVLine(line).map(v => v.replace(/^"|"$/g, '').trim())
          const obj = {}
          headers.forEach((header, idx) => {
            obj[header] = values[idx] || ''
          })
          return obj
        }).filter(row => Object.keys(row).length > 0 && Object.values(row).some(v => v))
      } catch (e) {
        console.error('Error parsing CSV:', e)
        return []
      }
    }
    return Array.isArray(csvData) ? csvData : []
  }, [csvData])

  // Get image paths - use uploaded images or fallback to default paths
  const images = React.useMemo(() => {
    if (imagePaths && imagePaths.length > 0) {
      return imagePaths.map(path => {
        // If it's a relative path, construct full URL
        if (path.startsWith('data/')) {
          return `http://localhost:8000/${path}`
        }
        return path
      })
    }
    // Fallback to default images
    return [
      'oral_scc_0018.jpg',
      'oral_scc_0045.jpg',
      'oral_scc_0059.jpg',
      'oral_scc_0094.jpg',
      'oral_scc_0103.jpg'
    ]
  }, [imagePaths])

  // Noise helper functions
  const hash = (n) => Math.abs(Math.sin(n) * 10000) % 1
  
  const noise = (x, y) => {
    const i = Math.floor(x)
    const j = Math.floor(y)
    const fx = x - i
    const fy = y - j
    const a = hash(i + j * 57)
    const b = hash(i + 1 + j * 57)
    const c = hash(i + (j + 1) * 57)
    const d = hash(i + 1 + (j + 1) * 57)
    const u = fx * fx * (3 - 2 * fx)
    const v = fy * fy * (3 - 2 * fy)
    return a * (1 - u) * (1 - v) + b * u * (1 - v) + c * (1 - u) * v + d * u * v
  }

  // Draw tumor on canvas
  const drawTumor = (size_cm, aggressiveness, nutritionFactor = 1.0) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    const container = canvas.parentElement
    if (!container) return
    
    // Get container dimensions
    const containerRect = container.getBoundingClientRect()
    const W = containerRect.width - 16 || 800 // Account for padding
    const H = 500 // Fixed height

    // Set canvas size accounting for device pixel ratio
    const ratio = window.devicePixelRatio || 1
    const displayWidth = W
    const displayHeight = H
    
    if (canvas.width !== displayWidth * ratio || canvas.height !== displayHeight * ratio) {
      canvas.width = displayWidth * ratio
      canvas.height = displayHeight * ratio
      canvas.style.width = `${displayWidth}px`
      canvas.style.height = `${displayHeight}px`
    }
    
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0)

    ctx.clearRect(0, 0, W, H)

    // Background vignette
    const g = ctx.createLinearGradient(0, 0, 0, H)
    g.addColorStop(0, 'rgba(2, 20, 35, 0.6)')
    g.addColorStop(1, 'rgba(1, 6, 12, 0.4)')
    ctx.fillStyle = g
    ctx.fillRect(0, 0, W, H)

    // Center point
    const cx = W * 0.48
    const cy = H * 0.52
    const baseR = 10 + size_cm * 18 * Math.sqrt(nutritionFactor)

    // Draw multiple overlapping irregular nodules
    const nodules = 25 + Math.floor(aggressiveness * 8)

    for (let n = 0; n < nodules; n++) {
      const angle = (n / nodules) * Math.PI * 2 + noise(n, size_cm) * 0.8
      const dist = baseR * (0.3 + 0.5 * noise(n * 2.3, size_cm * 1.7))
      const noduleX = cx + Math.cos(angle) * dist
      const noduleY = cy + Math.sin(angle) * dist
      const noduleR = baseR * (0.15 + 0.25 * noise(n * 3.1, size_cm * 2.3)) * (0.8 + aggressiveness * 0.15)

      // Each nodule is an irregular blob
      ctx.beginPath()
      const steps = 32
      for (let i = 0; i <= steps; i++) {
        const theta = (i / steps) * Math.PI * 2
        const noiseVal = noise(
          noduleX * 0.02 + Math.cos(theta) * 3,
          noduleY * 0.02 + Math.sin(theta) * 3
        )
        const r = noduleR * (0.6 + 0.5 * noiseVal)
        const x = noduleX + r * Math.cos(theta)
        const y = noduleY + r * Math.sin(theta)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.closePath()

      // Color each nodule with variation
      const brightness = 0.6 + 0.4 * noise(n * 4.7, size_cm)
      const redness = Math.floor(100 + 80 * brightness)
      const darkness = Math.floor(20 + 40 * (1 - brightness))
      ctx.fillStyle = `rgba(${redness}, ${darkness}, ${darkness}, 0.85)`
      ctx.fill()

      // Add subtle highlights
      if (n % 3 === 0) {
        ctx.strokeStyle = 'rgba(140, 50, 50, 0.3)'
        ctx.lineWidth = 1
        ctx.stroke()
      }
    }

    // Add darker patches for depth and necrotic areas
    for (let p = 0; p < 12; p++) {
      const angle = noise(p * 5.1, size_cm * 3.2) * Math.PI * 2
      const dist = baseR * (0.2 + 0.4 * noise(p * 2.7, size_cm * 1.9))
      const patchX = cx + Math.cos(angle) * dist
      const patchY = cy + Math.sin(angle) * dist
      const patchR = baseR * 0.2 * noise(p * 3.9, size_cm)

      const grad = ctx.createRadialGradient(patchX, patchY, 0, patchX, patchY, patchR)
      grad.addColorStop(0, 'rgba(60, 15, 15, 0.7)')
      grad.addColorStop(1, 'rgba(60, 15, 15, 0)')
      ctx.fillStyle = grad
      ctx.beginPath()
      ctx.arc(patchX, patchY, patchR, 0, Math.PI * 2)
      ctx.fill()
    }

    // Overall shadow for depth
    ctx.globalCompositeOperation = 'multiply'
    const shadowGrad = ctx.createRadialGradient(cx, cy + baseR * 0.3, baseR * 0.3, cx, cy, baseR * 1.5)
    shadowGrad.addColorStop(0, 'rgba(0, 0, 0, 0.3)')
    shadowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)')
    ctx.fillStyle = shadowGrad
    ctx.beginPath()
    ctx.arc(cx, cy, baseR * 1.3, 0, Math.PI * 2)
    ctx.fill()
    ctx.globalCompositeOperation = 'source-over'
  }

  // Update visualization for current index
  const updateForIndex = useCallback((i) => {
    if (!data || data.length === 0 || i < 0 || i >= data.length) return

    const row = data[i]
    const size = parseFloat(row['Tumor_Size_cm'] || row['tumor_size_cm'] || row['Tumor Size'] || '1.0') || 1.0
    const month = row['Follow_Up_Month'] || row['month_index'] || row['Follow-Up Month'] || i + 1
    const treatment = (row['Treatment_Type'] || row['treatment_type'] || row['Treatment Type'] || 'Unknown') + 
                     ' · ' + (row['Response_to_Treatment'] || row['response'] || row['Response to Treatment'] || 'Unknown')
    const recurrence = row['Recurrence'] || row['recurrence'] || ''
    const aggressiveness = recurrence && recurrence.toLowerCase().includes('yes') ? 2.6 : 1.4
    const smoking = row['Smoking_Status'] || row['smoking_status'] || row['Smoking Status'] || ''
    const nutrition = smoking && smoking.toLowerCase().includes('smoker') ? 0.85 : 1.0

    drawTumor(size, aggressiveness, nutrition)

    // Update image preview
    if (images.length > 0) {
      setCurrentImageIndex(i % images.length)
    }

    return { size, month, treatment, row }
  }, [data, images.length])

  // Handle play/pause
  const handlePlayPause = () => {
    if (isPlaying) {
      setIsPlaying(false)
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current)
        playIntervalRef.current = null
      }
    } else {
      setIsPlaying(true)
      playIntervalRef.current = setInterval(() => {
        setCurrentIndex(prev => {
          const next = (prev + 1) % data.length
          return next
        })
      }, 900)
    }
  }

  // Handle reset
  const handleReset = () => {
    setIsPlaying(false)
    if (playIntervalRef.current) {
      clearInterval(playIntervalRef.current)
      playIntervalRef.current = null
    }
    setCurrentIndex(0)
  }

  // Handle slider change
  const handleSliderChange = (e) => {
    const newIndex = parseInt(e.target.value)
    setCurrentIndex(newIndex)
  }

  // Update canvas when index or data changes
  useEffect(() => {
    if (data.length > 0 && currentIndex >= 0 && currentIndex < data.length) {
      // Use setTimeout to ensure canvas is rendered
      const timer = setTimeout(() => {
        updateForIndex(currentIndex)
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [currentIndex, updateForIndex])

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current)
      }
    }
  }, [])

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (data.length > 0 && currentIndex >= 0 && currentIndex < data.length) {
        updateForIndex(currentIndex)
      }
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [currentIndex, updateForIndex])

  if (!data || data.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Tumor Evolution Visualization</h3>
        <p className="text-gray-600">Please upload a CSV file to view the tumor evolution visualization.</p>
      </div>
    )
  }

  const currentRow = data[currentIndex] || {}
  const currentSize = parseFloat(currentRow['Tumor_Size_cm'] || currentRow['tumor_size_cm'] || currentRow['Tumor Size'] || '1.0') || 1.0
  const currentMonth = currentRow['Follow_Up_Month'] || currentRow['month_index'] || currentRow['Follow-Up Month'] || currentIndex + 1
  const currentTreatment = (currentRow['Treatment_Type'] || currentRow['treatment_type'] || currentRow['Treatment Type'] || 'Unknown') + 
                          ' · ' + (currentRow['Response_to_Treatment'] || currentRow['response'] || currentRow['Response to Treatment'] || 'Unknown')
  const patientId = currentRow['Patient_ID'] || currentRow['patient_id'] || currentRow['Patient ID'] || 'N/A'
  const age = currentRow['Age'] || currentRow['age'] || 'N/A'
  const stage = currentRow['Stage_TNM'] || currentRow['stage_tnm'] || currentRow['Stage'] || 'N/A'
  const comorbidities = currentRow['Comorbidities'] || currentRow['comorbidities'] || 'N/A'

  return (
    <div className="bg-white rounded-lg shadow p-6 mt-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Tumor Evolution Visualization</h3>
      <div className="flex flex-col lg:flex-row gap-4">
        {/* Controls Panel */}
        <div className="w-full lg:w-80 bg-gradient-to-b from-gray-50 to-gray-100 p-4 rounded-lg">
          <h4 className="font-semibold text-gray-800 mb-2">Patient Evolution</h4>
          <div className="text-xs text-gray-600 mb-4" id="metaBox">
            Patient ID: {patientId} · Age: {age} · Stage: {stage} · Comorbidities: {comorbidities}
          </div>
          
          <label className="block text-sm text-gray-700 mb-2">
            Follow-up month: <span className="font-medium">{currentMonth} month</span>
          </label>
          
          <input
            type="range"
            min="0"
            max={data.length - 1}
            step="1"
            value={currentIndex}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          
          <div className="flex gap-2 mt-4">
            <button
              onClick={handlePlayPause}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
            >
              {isPlaying ? 'Pause ❚❚' : 'Play ▶'}
            </button>
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
            >
              Reset
            </button>
          </div>
          
          {images.length > 0 && (
            <div className="mt-4">
              <label className="block text-sm text-gray-700 mb-2">Histology / specimen preview</label>
              <img
                src={images[currentImageIndex]}
                alt="Histology preview"
                className="w-full h-40 object-cover rounded-lg"
                onError={(e) => {
                  e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><rect width="200" height="200" fill="%23ddd"/><text x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%23999">No Image</text></svg>'
                }}
              />
            </div>
          )}
        </div>

        {/* Canvas Area */}
        <div className="flex-1 flex flex-col gap-3">
          <div className="bg-gradient-to-b from-slate-900 to-slate-800 rounded-lg p-2 shadow-lg" style={{ minHeight: '500px' }}>
            <canvas
              ref={canvasRef}
              className="w-full rounded-lg"
              style={{ 
                background: 'linear-gradient(180deg, #021423, #081827)',
                height: '500px',
                display: 'block'
              }}
            />
          </div>
          
          <div className="bg-gradient-to-b from-gray-50 to-gray-100 p-4 rounded-lg">
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
              <div className="flex-1">
                <strong className="block text-sm text-gray-700 mb-1">Tumor size (cm)</strong>
                <div className="text-2xl font-bold text-amber-600">{currentSize.toFixed(2)} cm</div>
              </div>
              <div className="w-full sm:w-56">
                <strong className="block text-sm text-gray-700 mb-1">Treatment / status</strong>
                <div className="text-sm text-green-700">{currentTreatment}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TumorEvolutionVisualization

