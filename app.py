import streamlit as st
import whisper
from datetime import timedelta
import os
import tempfile
from pydub import AudioSegment
# import torch

# Configuración de la página
st.set_page_config(page_title="Transcripción de Audio o Video", layout="wide")
st.title("📝 Transcripción de Audio o Video con Whisper")

MAX_FILE_SIZE = 40 * 1024 * 1024  # 30MB en bytes

# Sidebar con opciones
with st.sidebar:
    st.header("Configuración")
    
    selected_option = st.radio(
        "Tipo de tarea",
        ["Transcribir (mantiene el idioma original)", "Traducir (traduce de español a inglés)"]
    )

    task = "transcribe" if "Transcribir" in selected_option else "translate"

# Función para convertir archivos a formato WAV

@st.cache_resource
def load_local_model():
    return whisper.load_model("./model/tiny.pt")

def convert_to_wav(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        audio = AudioSegment.from_file(tmp_path)
        wav_path = os.path.splitext(tmp_path)[0] + ".wav"
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        st.error(f"Error al convertir el archivo: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Función para verificar tamaño de archivo
def validate_file_size(file):
    if file.size > MAX_FILE_SIZE:
        st.error(f"El archivo es demasiado grande. Tamaño máximo permitido: {MAX_FILE_SIZE/(1024*1024):.0f}MB")
        return False
    return True

# Interfaz principal
uploaded_file = st.file_uploader(
    label="**Sube un archivo de audio o video (máximo 30MB)**",
    type=["mp3", "wav", "ogg", "m4a", "mp4", "avi", "mov", "mkv"],
    )


def transcribe_large_audio(model, audio_path, task="transcribe"):
    whisper.load_model("./model/tiny.pt") 
    try:
        # Cargar el audio completo
        audio = whisper.load_audio(audio_path)
        
        # Opción 1: Transcribir todo de una vez (puede fallar con archivos muy grandes)
        result = model.transcribe(audio_path, task=task, language="es")
        
        return result
    except Exception as e:
        st.error(f"Error durante la transcripción: {e}")
        return None

# Función para generar archivo SRT
def generate_srt(segments, output_file):
    with open(output_file, 'w', encoding='utf-8') as srt_file:
        for segment in segments:
            start_time = str(timedelta(seconds=round(segment['start'], 3))).replace('.', ',')
            end_time = str(timedelta(seconds=round(segment['end'], 3))).replace('.', ',')
            text = segment['text'].strip()
            
            srt_file.write(f"{segment['id']+1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")

# Interfaz principal
if uploaded_file is not None:
    # Validar tamaño del archivo
    if not validate_file_size(uploaded_file):
        st.stop()  # Detener la ejecución si el archivo es muy grande
    
    # Mostrar información del archivo
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"📄 **Archivo:** {uploaded_file.name}")
    with col2:
        st.write(f"📊 **Tamaño:** {uploaded_file.size / (1024*1024):.2f} MB")
    
    # Convertir a WAV si es necesario
    with st.spinner("Preparando archivo para transcripción..."):
        audio_path = convert_to_wav(uploaded_file)
    
    if audio_path:
        # Cargar modelo (con cache para mejor rendimiento)
        model = whisper.load_model("./model/tiny.pt")
        
        # Transcribir
        with st.spinner("Transcribiendo contenido. Por favor, espera..."):
            result = model.transcribe(audio_path, task=task)

        if result:
            st.success("✅ Transcripción completada!")
            
            # Mostrar texto completo
            st.subheader("Texto completo")
            st.text_area("Transcripción", result["text"], height=200)
            
            # Mostrar segmentos con marcas de tiempo
            st.subheader("Segmentos con marcas de tiempo")
            for segment in result["segments"]:
                st.write(f"⏱️ {segment['start']:.1f}s - {segment['end']:.1f}s: {segment['text']}")
            
            # Generar y descargar SRT
            srt_filename = os.path.splitext(uploaded_file.name)[0] + ".srt"
            with tempfile.NamedTemporaryFile(mode='w+', suffix=".srt", delete=False) as tmp_srt:
                generate_srt(result["segments"], tmp_srt.name)
                
                with open(tmp_srt.name, "rb") as f:
                    st.download_button(
                        "⬇️ Descargar subtítulos (SRT)",
                        f,
                        file_name=srt_filename,
                        mime="text/plain"
                    )
            
            # Opción para descargar texto plano
            st.download_button(
                "⬇️ Descargar texto plano",
                result["text"],
                file_name=os.path.splitext(uploaded_file.name)[0] + ".txt",
                mime="text/plain"
            )
        
        # Limpiar archivos temporales
        if os.path.exists(audio_path):
            os.remove(audio_path)
else:
    st.info("👆 Por favor, sube un archivo de audio o video para comenzar")