import { useState } from 'react'

export default function Landpage() {
    const [image, setImage] = useState<File | null>(null);
    const [heatmap, setHeatmap] = useState<string | null>(null);

    const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setImage(e.target.files[0]);
        }
    }

    const sendToBackend = async () => {
        if (!image) return;

        const formData = new FormData();
        formData.append("file", image);

        const res = await fetch("http://localhost:5000/predict", {
            method: "POST",
            body: formData,
        });

        const data = await res.json();

        console.log(data);

        // Show heatmap
        const heatmapSrc = `data:image/jpeg;base64,${data.heatmap}`;
        setHeatmap(heatmapSrc);
    }

    return (
        <div>
            <input type='file' onChange={handleUpload} />
            <button onClick={sendToBackend}>Identify Area</button>

            {heatmap && (
                <img src={heatmap} alt="Heatmap" />
            )}
        </div>
    )
}