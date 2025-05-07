from gradio_client import Client, handle_file

client = Client("http://192.168.1.58:7860/")
result = client.predict(
		ref_audio_input=handle_file('/root/smart-yolo/MegaTTS3/assets/女/孤独感女声朗读.wav'),
		ref_text_input="但他的灯光温暖而柔和,每当夜晚降临,他便悄悄地亮起,照亮森林中的一条小路",
		gen_text_input="我是小明,我是一个程序员,我喜欢编程,我喜欢玩游戏,我喜欢看电影,我喜欢听音乐,我喜欢旅游,我喜欢美食,我喜欢睡觉,我喜欢学习,我喜欢生活,我喜欢你",
		remove_silence=False,
		randomize_seed=True,
		seed_input=0,
		cross_fade_duration_slider=0.15,
		nfe_slider=32,
		speed_slider=1,
		api_name="/basic_tts"
)
print(result)
