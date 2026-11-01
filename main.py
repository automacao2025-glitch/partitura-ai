import os
import shutil
import asyncio
import traceback
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import music21
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from basic_pitch.inference import predict

app = FastAPI(title="Partitura.ai - Universal Engine")

UPLOAD_DIR = Path("temp_uploads")
OUTPUT_DIR = Path("temp_outputs")
for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Partitura.ai - Partitura Inteligente</title>
    <script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.8.6/build/opensheetmusicdisplay.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #f0f4f8; color: #1e293b; margin: 0; padding: 25px; }
        .container { max-width: 980px; margin: 0 auto; background: #ffffff; padding: 35px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.06); }
        h1 { color: #0f172a; text-align: center; margin-bottom: 6px; font-size: 28px; }
        .subtitle { text-align: center; color: #64748b; margin-top: 0; margin-bottom: 28px; font-size: 15px; }
        
        .upload-box { border: 2px dashed #3b82f6; padding: 40px 20px; text-align: center; border-radius: 12px; background: #f8fafc; cursor: pointer; transition: all 0.25s ease; margin-bottom: 20px; }
        .upload-box:hover { background: #eff6ff; border-color: #2563eb; }
        
        #fileStatus { display: none; background: #dcfce7; color: #166534; padding: 14px; border-radius: 8px; border: 1px solid #bbf7d0; margin-bottom: 20px; text-align: center; font-weight: 600; font-size: 15px; }
        
        button.btn-convert { background: #2563eb; color: white; border: none; padding: 16px 24px; font-size: 17px; border-radius: 10px; cursor: pointer; width: 100%; font-weight: bold; transition: 0.2s; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2); }
        button.btn-convert:hover { background: #1d4ed8; }
        button.btn-convert:disabled { background: #94a3b8; cursor: not-allowed; box-shadow: none; }
        
        #loadingSection { display: none; text-align: center; margin-top: 24px; padding: 24px; background: #e0f2fe; border: 1px solid #bae6fd; border-radius: 10px; color: #0369a1; }
        .spinner { display: inline-block; width: 28px; height: 28px; border: 4px solid #7dd3fc; border-top-color: #0284c7; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 10px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        #resultSection { display: none; margin-top: 32px; border-top: 2px solid #f1f5f9; padding-top: 24px; }
        .ai-telemetry { background: #f8fafc; border: 1px solid #e2e8f0; color: #0f172a; padding: 14px 20px; border-radius: 10px; text-align: center; font-weight: 600; font-size: 15px; margin-bottom: 20px; }
        
        .action-bar { display: flex; justify-content: center; gap: 16px; margin-bottom: 25px; flex-wrap: wrap; }
        .btn-midi { background: #10b981; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; transition: 0.2s; }
        .btn-midi:hover { background: #059669; }
        .btn-pdf { background: #ef4444; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 15px; transition: 0.2s; }
        .btn-pdf:hover { background: #dc2626; }
        
        #sheetMusicContainer { width: 100%; min-height: 450px; background: white; padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px; box-sizing: border-box; overflow-x: auto; }

        @media print {
            body { background: white; padding: 0; margin: 0; }
            .container { box-shadow: none; max-width: 100%; width: 100%; padding: 0; margin: 0; }
            h1, .subtitle, form, #loadingSection, .action-bar, .ai-telemetry, #resultTitle, #previewTitle { display: none !important; }
            #resultSection { border: none; margin: 0; padding: 0; display: block !important; }
            #sheetMusicContainer { border: none; padding: 0; margin: 0; width: 100%; overflow: visible; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Partitura.ai</h1>
        <p class="subtitle">Transcrição Musical Inteligente, Rápida e Autónoma</p>
        
        <form id="uploadForm">
            <div class="upload-box" onclick="document.getElementById('audioFile').click()">
                <strong style="font-size: 17px; color: #1e293b;">Clique aqui ou arraste o seu ficheiro de áudio</strong><br>
                <span style="font-size: 13px; color: #64748b; display: inline-block; margin-top: 6px;">Formatos suportados: MP3, WAV, M4A</span>
                <input type="file" id="audioFile" name="file" accept=".mp3,.m4a,.wav" style="display: none;" required>
            </div>
            
            <div id="fileStatus">
                ✅ Ficheiro "<span id="fileNameText"></span>" pronto!
            </div>

            <button type="submit" id="submitBtn" class="btn-convert" disabled>✨ Gerar Partitura Exata</button>
        </form>

        <div id="loadingSection">
            <div class="spinner"></div>
            <h3 style="margin: 0 0 6px 0;">⚡ A processar a IA...</h3>
            <p id="stepText" style="margin: 0; font-size: 14px;">A converter formato e a compor pauta musical em Clave de Sol...</p>
        </div>

        <div id="resultSection">
            <h2 id="resultTitle" style="color: #10b981; text-align: center; margin-top: 0;">✨ Partitura Gerada com Sucesso!</h2>
            
            <div id="telemetryBanner" class="ai-telemetry">
                🎵 Detectado: -- BPM | Compasso: -- | Tom: -- | Clave: --
            </div>
            
            <div class="action-bar">
                <a id="downloadMidiBtn" href="#" download class="btn-midi">📥 Baixar Ficheiro MIDI</a>
                <button type="button" class="btn-pdf" id="savePdfBtn">📄 Salvar / Imprimir Partitura em PDF</button>
            </div>

            <h3 id="previewTitle" style="margin-top: 20px; text-align: center; color: #334155;">📖 Lead Sheet Definitivo:</h3>
            <div id="sheetMusicContainer"></div>
        </div>
    </div>

    <script>
        const audioInput = document.getElementById('audioFile');
        const fileStatus = document.getElementById('fileStatus');
        const fileNameText = document.getElementById('fileNameText');
        const form = document.getElementById('uploadForm');
        const submitBtn = document.getElementById('submitBtn');
        const loadingSection = document.getElementById('loadingSection');
        const resultSection = document.getElementById('resultSection');
        const telemetryBanner = document.getElementById('telemetryBanner');
        const savePdfBtn = document.getElementById('savePdfBtn');
        let osmd = null;

        audioInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                fileNameText.innerText = e.target.files[0].name;
                fileStatus.style.display = 'block';
                submitBtn.disabled = false;
            } else {
                fileStatus.style.display = 'none';
                submitBtn.disabled = true;
            }
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            
            resultSection.style.display = 'none';
            loadingSection.style.display = 'block';
            submitBtn.disabled = true;
            submitBtn.innerText = "A compor pauta...";

            try {
                const response = await fetch('/convert', { method: 'POST', body: formData });
                const data = await response.json();

                if (response.ok) {
                    loadingSection.style.display = 'none';
                    resultSection.style.display = 'block';
                    
                    telemetryBanner.innerText = `🎵 Ritmo: ${data.bpm} BPM | Compasso: ${data.time_signature} | Tom: ${data.key_name} | Clave: ${data.clef_name}`;
                    document.getElementById('downloadMidiBtn').href = data.midi_url;

                    const container = document.getElementById('sheetMusicContainer');
                    container.innerHTML = '';
                    osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
                        autoResize: true,
                        backend: "svg",
                        drawTitle: true,
                        drawPartNames: false
                    });

                    await osmd.load(data.xml_content);
                    osmd.render();
                } else {
                    alert("Aviso: " + (data.detail || "Falha no processamento."));
                    loadingSection.style.display = 'none';
                }
            } catch (err) {
                alert("Erro de comunicação com o servidor: " + err.message);
                loadingSection.style.display = 'none';
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerText = "✨ Gerar Partitura Exata";
            }
        });

        savePdfBtn.addEventListener('click', () => {
            window.print();
        });
    </script>
</body>
</html>
"""

def fast_audio_processing(safe_input_path: Path):
    clean_target = UPLOAD_DIR / "normalized_input.wav"
    try:
        try:
            y, sr = librosa.load(str(safe_input_path), sr=22050, mono=True)
        except Exception:
            try:
                data, samplerate = sf.read(str(safe_input_path))
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                if samplerate != 22050:
                    import scipy.signal
                    num_samples = int(len(data) * 22050 / samplerate)
                    data = scipy.signal.resample(data, num_samples)
                y, sr = data.astype(np.float32), 22050
            except Exception as e2:
                raise RuntimeError(f"Não foi possível ler este formato de áudio. Tente enviar um ficheiro .MP3 ou .WAV. Erro técnico: {str(e2)}")
        
        sf.write(str(clean_target), y, sr, subtype='PCM_16')

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_result = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo = tempo_result[0] if isinstance(tempo_result, tuple) else tempo_result
        
        bpm = float(np.ndim(tempo) and tempo[0] or tempo)
        if bpm <= 0 or np.isnan(bpm): bpm = 120.0
        if bpm > 170: bpm /= 2.0
        elif bpm < 55: bpm *= 2.0
        
        time_sig = "2/4" if bpm > 130 else "4/4"

        _, midi_data, _ = predict(str(clean_target.resolve()))
        return midi_data, str(clean_target), round(bpm, 1), time_sig
    except RuntimeError as re:
        raise re
    except Exception as e:
        raise RuntimeError(f"Erro no pipeline rápido: {type(e).__name__} - {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.post("/convert")
async def convert_audio_ai(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    safe_input_path = UPLOAD_DIR / f"raw_upload{ext}"
    
    try:
        with open(safe_input_path, "wb") as f:
            f.write(await file.read())

        midi_data, clean_harm_path, detected_bpm, detected_ts = await asyncio.to_thread(
            fast_audio_processing, safe_input_path
        )

        sec_per_quarter = 60.0 / detected_bpm
        beats_per_bar = int(detected_ts.split("/")[0])
        grid_step = sec_per_quarter / 4.0 

        all_notes = []
        for inst in midi_data.instruments:
            for n in inst.notes:
                if (n.end - n.start) >= 0.08 and n.velocity >= 35:
                    
                    if n.pitch < 57: # 57 = A3 (Lá3)
                        n.pitch += 12 
                    if n.pitch < 57: 
                        n.pitch += 12 
                        
                    n.start = round(n.start / grid_step) * grid_step
                    n.end = round(n.end / grid_step) * grid_step
                    
                    if n.end <= n.start:
                        n.end = n.start + grid_step
                    all_notes.append(n)

        if not all_notes:
            raise HTTPException(status_code=422, detail="Nenhuma nota musical clara detetada no áudio.")

        midi_filename = "partitura_exata.mid"
        midi_path = OUTPUT_DIR / midi_filename
        midi_data.write(str(midi_path))

        part = music21.stream.Part()
        part.insert(0, music21.clef.TrebleClef())
        part.insert(0, music21.meter.TimeSignature(detected_ts))
        part.insert(0, music21.tempo.MetronomeMark(number=detected_bpm))

        notes_sorted = sorted(all_notes, key=lambda x: x.start)
        clusters = []
        for n in notes_sorted:
            if not clusters or (n.start - clusters[-1][0].start) > 0.05:
                clusters.append([n])
            else:
                clusters[-1].append(n)

        for idx, cluster in enumerate(clusters):
            raw_start = cluster[0].start / sec_per_quarter
            raw_dur = max(c.end for c in cluster) / sec_per_quarter - raw_start
            
            if idx + 1 < len(clusters):
                next_start = clusters[idx + 1][0].start / sec_per_quarter
                if raw_start + raw_dur > next_start:
                    raw_dur = next_start - raw_start
            
            start_ql = round(raw_start * 4) / 4.0
            dur_ql = round(raw_dur * 4) / 4.0
            
            if dur_ql < 0.25: dur_ql = 0.25
            if dur_ql > beats_per_bar: dur_ql = float(beats_per_bar)
            
            pitches_in_cluster = sorted(list({c.pitch for c in cluster}))[-2:]
            
            if len(pitches_in_cluster) == 1:
                el = music21.note.Note()
                el.pitch.midi = int(pitches_in_cluster[0])
            else:
                notas_acorde = []
                for p in pitches_in_cluster:
                    n_obj = music21.note.Note()
                    n_obj.pitch.midi = int(p)
                    notas_acorde.append(n_obj)
                el = music21.chord.Chord(notas_acorde)
            
            el.quarterLength = dur_ql
            part.insert(start_ql, el)

        part.quantize(quarterLengthDivisors=(2, 4), processOffsets=True, processDurations=True, inPlace=True)

        ks = music21.analysis.discrete.KrumhanslSchmuckler()
        detected_key = ks.getSolution(part)
        key_sig = music21.key.KeySignature(detected_key.sharps)
        part.insert(0, key_sig)

        tonica_map = {'C': 'Dó', 'G': 'Sol', 'D': 'Ré', 'A': 'Lá', 'E': 'Mi', 'B': 'Si', 'F#': 'Fá#', 'F': 'Fá', 'Bb': 'Sib', 'Eb': 'Mib', 'Ab': 'Láb'}
        tonic_pt = tonica_map.get(detected_key.tonic.name, detected_key.tonic.name)
        mode_pt = "Maior" if detected_key.mode == "major" else "Menor"
        key_label = f"{tonic_pt} {mode_pt}"

        final_score = music21.stream.Score()
        final_score.metadata = music21.metadata.Metadata()
        final_score.metadata.title = os.path.splitext(file.filename)[0]
        final_score.metadata.composer = f"Partitura.ai ({detected_bpm} BPM)"

        measured_part = part.makeMeasures(inPlace=False)
        measured_part.makeTies(inPlace=True)
        measured_part.makeBeams(inPlace=True)
        measured_part.makeNotation(inPlace=True)
        
        final_score.append(measured_part)

        xml_path = OUTPUT_DIR / "partitura_exata.musicxml"
        final_score.write('musicxml', fp=str(xml_path))

        with open(xml_path, "r", encoding="utf-8") as xf:
            xml_content = xf.read()

        return {
            "midi_url": f"/download/{midi_filename}",
            "xml_content": xml_content,
            "bpm": detected_bpm,
            "time_signature": detected_ts,
            "key_name": key_label,
            "clef_name": "Clave de Sol"
        }

    except Exception as e:
        erro_completo = traceback.format_exc()
        print("ERRO DETALHADO NO BACKEND:")
        print(erro_completo)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in UPLOAD_DIR.glob("*"):
            try: p.unlink()
            except Exception: pass

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    return FileResponse(str(file_path), media_type='audio/midi', filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8005, reload=True)