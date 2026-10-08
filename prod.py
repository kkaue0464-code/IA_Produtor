import streamlit as st
import librosa
import librosa.display
import numpy as np
import subprocess
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    # Carregar o áudio com Librosa
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

    # --- ABA 2: VISUALIZAÇÃO INTERATIVA COM CROSSHAIR ---
    st.subheader("🎨 Visualização Interativa de Frequência e Forma de Onda")
    st.write("💡 *Passe o mouse ou clique sobre os gráficos para usar o Crosshair e analisar tempo/frequência exatos.*")

    # Dados para a Waveform
    time_wave = np.linspace(0, duration, num=len(y))

    # Dados para o Espectrograma (STFT)
    D = np.abs(librosa.stft(y))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr)
    time_spec = librosa.frames_to_time(np.arange(D_db.shape[1]), sr=sr)

    # Subplots do Plotly
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("Forma de Onda (Amplitude x Tempo)", "Espectrograma (Frequência em Hz x Tempo)")
    )

    # 1. Waveform
    fig.add_trace(
        go.Scatter(x=time_wave[::10], y=y[::10], mode='lines', name='Waveform', line=dict(color='#8A2BE2', width=1)),
        row=1, col=1
    )

    # 2. Espectrograma
    fig.add_trace(
        go.Heatmap(
            z=D_db,
            x=time_spec,
            y=freqs,
            colorscale='Magma',
            colorbar=dict(title="dB", len=0.45, y=0.2),
            hovertemplate="Tempo: %{x:.2f}s<br>Freq: %{y:.0f} Hz<br>Energia: %{z:.1f} dB<extra></extra>"
        ),
        row=2, col=1
    )

    # Configuração do Crosshair
    fig.update_layout(
        height=600,
        showlegend=False,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash', spikecolor='cyan', spikethickness=1, title_text="Tempo (segundos)", row=2, col=1)
    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dash', spikecolor='cyan', spikethickness=1)

    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Frequência (Hz)", range=[0, min(sr//2, 20000)], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # --- ABA 3: DIAGNÓSTICO DO PRODUTOR (DIREÇÃO CRIATIVA) ---
    st.header("💡 Dicas do Agente para sua Mix/Gravação")
    
    if bpm < 90:
        st.info("📌 **Dica de Arranjo:** Sua track tem uma pegada mais *Slow Jam*, *Hip Hop OldSchool* ou *Doom Metal*. Cuidado com o acúmulo de sub-graves no bumbo (kick) e no baixo para não embolar a mix.")
    elif 90 <= bpm <= 128:
        st.info("📌 **Dica de Arranjo:** Ritmo comercial padrão (Pop, House, Groove). Foque no transiente da caixa (snare) na casa dos 200Hz para dar aquele 'soco' que corta a mix.")
    else:
        st.info("📌 **Dica de Arranjo:** Andamento acelerado (Drum & Bass, Techno, Rock). Atenção aos elementos de alta frequência (hi-hats, pratos). Use um de-esser ou corte leve em 10kHz se o brilho estiver machucando o ouvido.")

    # --- ABA 4: SEPARAÇÃO DE TRILHAS EM 4 FAIXAS (DEMUCS) ---
    st.header("🪓 Separação de Stems (Voz, Bateria, Baixo e Outros)")
    st.write("Isole os elementos individuais da sua track em 4 faixas separadas com o Demucs.")
    
    if st.button("Separar Áudio em 4 Tracks (Vocais, Drums, Bass, Outros)"):
        with st.spinner("O Demucs está separando as faixas... Aguarde!"):
            try:
                command = f'"{sys.executable}" -m demucs.separate -o output temp_audio.wav'
                subprocess.run(command, shell=True, check=True)
                
                st.success("Pronto! Áudio fatiado em 4 faixas com sucesso.")
                
                base_path = "output/htdemucs/temp_audio"
                
                if os.path.exists(base_path):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.audio(f"{base_path}/vocals.wav", format="audio/wav")
                        st.caption("🎙️ Vocais (Acapella)")
                        
                        st.audio(f"{base_path}/drums.wav", format="audio/wav")
                        st.caption("🥁 Bateria / Percussão")
                        
                    with col_b:
                        st.audio(f"{base_path}/bass.wav", format="audio/wav")
                        st.caption("🎸 Baixo / Sub")
                        
                        st.audio(f"{base_path}/other.wav", format="audio/wav")
                        st.caption("🎹 Outros (Teclados, Guitarras, Synths)")
                else:
                    st.error("Arquivo processado não foi encontrado na pasta de saída.")
                    
            except Exception as e:
                st.error(f"Erro ao rodar o Demucs: {e}")

else:
    st.warning("👋 Alô, produtor! Pendura um arquivo de áudio ali na barra lateral para a gente começar o Directing.")
