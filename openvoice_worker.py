import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import argparse

def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt_base = f"checkpoints/base_speakers/EN"
    ckpt_converter = f"checkpoints/converter"

    source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    return base_speaker_tts, tone_color_converter, source_se, device

def synthesize(ref_audio_path, text, output_dir):

    base_speaker_tts, tone_color_converter, source_se, device = load_model()


    target_se, audio_name = se_extractor.get_se(ref_audio_path, tone_color_converter, target_dir='processed', vad=True)
    src_path = f'./tmp.wav'
    base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.0)

    encode_message = "@MyShell"
    tone_color_converter.convert(
    audio_src_path=src_path,
    src_se=source_se,
    tgt_se=target_se,
    output_path=output_dir,
    message=encode_message)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVoice CLI TTS")
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio file")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    synthesize(args.ref_audio, args.text, args.output_dir)