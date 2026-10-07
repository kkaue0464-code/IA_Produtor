import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import soundfile as sf
# Nota: O Spleeter geralmente é chamado via CLI ou subprocess no ambiente virtual devido às dependências do TensorFlow
import subprocess


# --- CONFIGURAÇÃO DA INTERFACE (STREAMLIT) ---
st.set_page_config(page_title="ProducerAI - Seu Assistente de Estúdio", page_icon="🎚️", layout="wide")

st.title("🎚️ ProducerAI: O Agente do Seu Estúdio")
st.markdown("""
    Bem-vindo ao front-desk da sua produção! Suba o seu arquivo de áudio (WAV ou MP3) 
    para analisarmos o **BPM, Tom, Balanço de Frequências** e dar aquele tapa na sua mix.
""")

# --- SIDEBAR: Upload do Arquivo ---
st.sidebar.header("🗂️ Input de Áudio")
uploaded_file = st.sidebar.file_uploader("Escolha a sua track", type=["wav", "mp3"])

if uploaded_file is not None:
    # Salvar temporariamente o arquivo para manipulação
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Carregar o áudio com Librosa"
    # sr=None mantém a taxa de amostragem original da gravação
    y, sr = librosa.load("temp_audio.wav", sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Playback nativo no Streamlit
    st.audio("temp_audio.wav", format="audio/wav")
    
    # --- ABA 1: ANÁLISE TÉCNICA (O "RAIO-X" DO ÁUDIO) ---
    st.header("📊 Análise Métrica da Track")
    col1, col2, col3 = st.columns(3)
    
    # 1. Cálculo de BPM (Tempo)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
    col1.metric(label="⏱️ BPM Estimado", value=f"{bpm} BPM")
    
    # 2. Estimativa de Tom (Key Detection básica via Cromagrama)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    estimated_key = notes[np.argmax(mean_chroma)]
    col2.metric(label="🎵 Tom Principal Estimado", value=estimated_key)
    
    # 3. Taxa de Amostragem e Duração
    col3.metric(label="⏳ Duração / Sample Rate", value=f"{round(duration, 2)}s @ {sr/1000}kHz")

    # --- ABA 2: VISUALIZAÇÃO CRATIVA (O ESPELHO DA MIX) ---
    st.subheader("🎨 Visualização de Frequência e Forma de Onda")
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Waveform (Forma de onda clássica)
    librosa.display.waveshow(y, sr=sr, ax=ax[0], color="purple")
    ax[0].set(title="Forma de Onda (Dinâmica e Transientes)")
    ax[0].label_outer()
    
    # Espectrograma de Frequências (Ver onde o som tá "pesando")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[1], cmap='magma')
    ax[1].set(title="Espectrograma (Análise de Frequências)")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    
    st.pyplot(fig)

    # --- ABA 3: DIAGNÓSTICO DO PRODUTOR (DIREÇÃO CRIATIVA) ---
    st.header("💡 Dicas do Agente para sua Mix/Gravação")
    
    # Lógica simples de tomada de decisão baseada nos dados extraídos
    if bpm < 90:
        st.info("📌 **Dica de Arranjo:** Sua track tem uma pegada mais *Slow Jam*, *Hip Hop OldSchool* ou *Doom Metal*. Cuidado com o acúmulo de sub-graves no bumbo (kick) e no baixo para não embolar a mix.")
    elif 90 <= bpm <= 128:
        st.info("📌 **Dica de Arranjo:** Ritmo comercial padrão (Pop, House, Groove). Foque no transiente da caixa (snare) na casa dos 200Hz para dar aquele 'soco' que corta a mix.")
    else:
        st.info("📌 **Dica de Arranjo:** Andamento acelerado (Drum & Bass, Techno, Rock). Atenção aos elementos de alta frequência (hi-hats, pratos). Use um de-esser ou corte leve em 10kHz se o brilho estiver machucando o ouvido.")

    # --- ABA 4: SEPARAÇÃO DE TRILHAS (SPLEETER) ---
    st.header("🪓 Separação de Stems (Voz e Instrumentos)")
    st.write("Quer isolar a acapella ou o instrumental para estudar o arranjo? O Spleeter resolve.")
    
    if st.button("Separar Áudio em 2 Tracks (Voz + Acompanhamento)"):
        with st.spinner("O Spleeter está trabalhando na mesa de som... Guenta aí!"):
            # Executa o comando de separação do spleeter via terminal
            try:
                command = "spleeter separate -p spleeter:2stems -o output/ temp_audio.wav"
                subprocess.run(command, shell=True, check=True)
                st.success("Pronto! Áudio fatiado com sucesso.")
                
                # Exibe os resultados salvos na pasta de output
                st.audio("output/temp_audio/vocals.wav", format="audio/wav")
                st.caption("🎙️ Vocais Isolados (Acapella)")
                st.audio("output/temp_audio/accompaniment.wav", format="audio/wav")
                st.caption("🎸 Instrumental / Beat")
            except Exception as e:
                st.error(f"Erro ao rodar o Spleeter: {e}. Certifique-se de que o modelo e as dependências do TensorFlow estão configurados no ambiente.")

else:
    st.warning("👋 Alô, produtor! Pendura um arquivo de áudio ali na barra lateral para a gente começar o Directing.")
