from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
output_audio_path, _, _, _ = client.predict(
		ref_audio_input=handle_file('/root/smart-yolo/MegaTTS3/assets/女/慵懒妩媚性感女声.wav'),
		ref_text_input="",
		gen_text_input="我想你了，想和你一起去旅行",
		remove_silence=False,
		randomize_seed=True,
		seed_input=0,
		cross_fade_duration_slider=0.15,
		nfe_slider=32,
		speed_slider=1,
		api_name="/basic_tts"
)
print(output_audio_path)
