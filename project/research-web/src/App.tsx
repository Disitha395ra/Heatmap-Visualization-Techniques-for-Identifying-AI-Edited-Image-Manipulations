import './App.css'
import Landpage from './components/Landpage'

function App() {
  return (
    <div className="app-container">
      <header className="header animate-fade-in">
        <h1>DeepFake Detection</h1>
        <p>Advanced image manipulation detection using ResNet50 & Grad-CAM</p>
      </header>
      <main>
        <Landpage />
      </main>
    </div>
  )
}

export default App
