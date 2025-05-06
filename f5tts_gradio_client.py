from gradio_client import Client, handle_file

client = Client("http://192.168.1.58:7860/")
result = client.predict(
		ref_audio_input=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),
		ref_text_input="Hello!!",
		gen_text_input="Hello!!",
		remove_silence=False,
		randomize_seed=True,
		seed_input=0,
		cross_fade_duration_slider=0.15,
		nfe_slider=32,
		speed_slider=1,
		api_name="/basic_tts"
)
print(result)
