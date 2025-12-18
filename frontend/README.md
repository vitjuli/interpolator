# Interpoletor Frontend

Modern, responsive Next.js frontend for the 5D Regression Neural Network System.

## Features

### 🎯 Single-Page Application
All functionality integrated into one comprehensive interface:

1. **Dataset Upload Section**
   - File upload with drag & drop support
   - .pkl file validation
   - File preview with metadata
   - Real-time upload status

2. **Training Configuration**
   - Dynamic hidden layer architecture builder
   - Configurable hyperparameters (epochs, learning rate)
   - Real-time training status
   - Training metrics display

3. **Prediction Interface**
   - 5 input fields for feature values
   - Real-time prediction results
   - Visual result display
   - Input validation

4. **Training History Visualization**
   - Interactive line charts for loss metrics
   - Training vs validation loss comparison
   - Final metrics summary

5. **System Information Dashboard**
   - Backend connection status monitor
   - Model loading status
   - Feature highlights
   - Performance information

### 🎨 Academic Design
- Clean, professional interface
- Academic color scheme (navy blue, gold accents)
- Responsive grid layout
- Smooth animations and transitions
- Comprehensive tooltips and help text

### 🔌 Backend Integration
- Automatic health check monitoring
- RESTful API integration with FastAPI backend
- Error handling and user feedback
- CORS-enabled communication

## Technology Stack

- **Framework**: Next.js 14 (React 18)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React
- **HTTP Client**: Axios

## Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn
- Backend server running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Development

The application will be available at `http://localhost:3000`

### Project Structure

```
frontend/
├── src/
│   └── app/
│       ├── globals.css       # Global styles and Tailwind
│       ├── layout.tsx         # Root layout
│       └── page.tsx           # Main single-page application
├── public/                    # Static assets
├── package.json              # Dependencies
├── tsconfig.json             # TypeScript configuration
├── tailwind.config.js        # Tailwind CSS configuration
└── next.config.js            # Next.js configuration
```

## API Endpoints Used

- `GET /health` - Backend health check
- `POST /upload` - Upload .pkl dataset
- `POST /train` - Train model with hyperparameters
- `POST /predict` - Make predictions

## Usage Guide

### 1. Upload Dataset
- Click the upload area or drag & drop a .pkl file
- File must contain 'X' (n_samples, 5) and 'y' (n_samples,) keys
- Preview shows file metadata
- Click "Upload Dataset" to send to backend

### 2. Configure Training
- Adjust hidden layer architecture by adding/removing layers
- Set max epochs (default: 200)
- Set learning rate (default: 0.001)
- Click "Start Training" to begin
- View results and metrics after completion

### 3. Make Predictions
- Enter 5 feature values (x₁ through x₅)
- Click "Predict" to get result
- Result displays with input summary
- Requires trained model to be loaded

### 4. Monitor System
- Backend status indicator in header (green = connected)
- Model status shows if model is loaded
- Real-time health checks every 10 seconds
- Training history chart appears after successful training

## Features Highlight

### Backend Status Monitor
- **Green (Connected)**: Backend is responsive
- **Red (Disconnected)**: Backend is not accessible
- **Yellow (Checking)**: Health check in progress

### Model Status
- **Blue (Loaded)**: Model is trained and ready
- **Gray (Not Loaded)**: No model available for predictions

### Responsive Design
- Fully responsive layout
- Mobile-friendly interface
- Adaptive grid system
- Touch-friendly controls

### Error Handling
- Comprehensive error messages
- User-friendly alerts
- Validation feedback
- Connection status updates

## Styling

The application uses a custom academic theme with:
- Primary: Navy Blue (#1e3a8a)
- Accent: Gold (#f59e0b)
- Background: Light gradient (slate → blue → indigo)
- Custom components with hover effects
- Smooth animations and transitions

## Troubleshooting

### Backend Connection Issues
- Ensure backend is running on `http://localhost:8000`
- Check CORS configuration in backend
- Verify .env.local has correct API URL

### File Upload Issues
- Ensure file is .pkl format
- Check file contains required 'X' and 'y' keys
- Verify backend upload endpoint is accessible

### Training Issues
- Upload dataset before training
- Check hyperparameter values are valid
- Ensure backend has sufficient resources

### Prediction Issues
- Train model before making predictions
- Ensure all 5 input fields are filled
- Check model is loaded (blue status)

## Contributing

This is an academic project for the Cambridge Computer Science Tripos.

## License

Academic project - Cambridge University

---

**Author**: Iuliia Vituigova (iv294@cam.ac.uk)
**Course**: Computer Science Tripos, Part I
**Year**: 2025
