import os
import random 
import gradio as gr
import numpy as np
import time
import torch, torchaudio
import gc
import warnings
warnings.filterwarnings('ignore')
from zhconv import convert
from LLM import LLM
from TTS import EdgeTTS
from src.cost_time import calculate_time
from configs import *
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

os.environ["GRADIO_TEMP_DIR"]= './temp'
os.environ["WEBUI"] = "true"

def get_title(title = 'Linly 智能对话系统 (Linly-Talker)'):
    description = f"""
    <p style="text-align: center; font-weight: bold;">
        <span style="font-size: 28px;">{title}</span>
        <br>
        <span style="font-size: 18px;" id="paper-info">
            [<a href="https://zhuanlan.zhihu.com/p/671006998" target="_blank">知乎</a>]
            [<a href="https://www.bilibili.com/video/BV1rN4y1a76x/" target="_blank">bilibili</a>]
            [<a href="https://github.com/Kedreamix/Linly-Talker" target="_blank">GitHub</a>]
            [<a herf="https://kedreamix.github.io/" target="_blank">个人主页</a>]
        </span>
        <br> 
        <span>Linly-Talker是一款创新的数字人对话系统，它融合了最新的人工智能技术，包括大型语言模型（LLM）🤖、自动语音识别（ASR）🎙️、文本到语音转换（TTS）🗣️和语音克隆技术🎤。</span>
    </p>
    """
    return description

# Default system and prompt settings
DEFAULT_SYSTEM = '你是一个很有帮助的助手'
PREFIX_PROMPT = '请用少于25个字回答以下问题\n\n'
# Default parameters
IMAGE_SIZE = 256
PREPROCESS_TYPE = 'crop'
FACERENDER = 'facevid2vid'
ENHANCER = False
IS_STILL_MODE = False
EXP_WEIGHT = 1
USE_REF_VIDEO = False
REF_VIDEO = None
REF_INFO = 'pose'
USE_IDLE_MODE = False
AUDIO_LENGTH = 5

edgetts = EdgeTTS()

@calculate_time
def Asr(audio):
    try:
        question = asr.transcribe(audio)
        question = convert(question, 'zh-cn')
    except Exception as e:
        gr.Warning("ASR Error: ", e)
        question = 'Gradio存在一些bug，麦克风模式有时候可能音频还未传入，请重新点击一下语音识别即可'
    return question

def clear_memory():
    """
    清理PyTorch的显存和系统内存缓存。
    """
    # 1. 清理缓存的变量
    gc.collect()  # 触发Python垃圾回收
    torch.cuda.empty_cache()  # 清理PyTorch的显存缓存
    torch.cuda.ipc_collect()  # 清理PyTorch的跨进程通信缓存
    
    # 2. 打印显存使用情况（可选）
    print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    print(f"Max cached memory: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB")

def generate_seed():
    seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": seed}

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def change_instruction(mode):
    return instruct_dict.get(mode, '未知模式')

PROMPT_SR, TARGET_SR = 16000, 22050
DEFAULT_DATA = np.zeros(TARGET_SR)

@calculate_time
def TTS_response(text, voice, rate, volume, pitch,
                tts_method='Edge-TTS', save_path='answer.wav'):
    if text == '':
        text = '请输入文字/问题'
    if tts_method == 'Edge-TTS':
        if not edgetts.network:
            gr.Warning("请检查网络或使用其他模型，例如PaddleTTS")
            return None
        try:
            edgetts.predict(text, voice, rate, volume, pitch, save_path, 'answer.vtt')
        except Exception as e:
            os.system(f'edge-tts --text "{text}" --voice {voice} --write-media {save_path} --write-subtitles answer.vtt')
        return save_path
    else:
        gr.Warning('未知模型')
    return None

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}


@calculate_time
def LLM_response(
    question,  # 输入的音频和文本问题
    voice,  # 语音合成参数
    tts_method='Edge-TTS'  # TTS 方法，默认使用 'Edge-TTS'
):
    if len(question) == 0:
        gr.Warning("请输入问题")
        return None, None, None

    # 生成回答
    answer = llm.generate(question, DEFAULT_SYSTEM)
    print("LLM 回复：", answer)

    # 合成回答语音
    tts_audio = TTS_response(
        answer, voice, 0, 100, 0
    )

    # 生成VTT文件（如果TTS方法为'Edge-TTS'）
    tts_vtt = 'answer.vtt' if tts_method == 'Edge-TTS' else None

    return tts_audio, tts_vtt, answer

@calculate_time
def Talker_response_img(text):


    driven_audio, driven_vtt, _ = LLM_response(text, 'zh-CN-XiaoxiaoNeural',
                                               'Edge-TTS')

    if driven_audio is None:
        gr.Warning("音频没有正常生成，请检查TTS是否正确")
        return None

    # 视频生成
    #video = None
    video = talker.test2('/root/autodl-tmp/Linly-Talker/myface.png', driven_audio, 'crop', False, False,
                             2, 256, 0, True, 1,
                             REF_VIDEO, REF_INFO, USE_IDLE_MODE, AUDIO_LENGTH, True, 
                             fps=20)
    """ if method == 'SadTalker':
        video = talker.test2(source_image, driven_audio, 'crop', False, False,
                             2, 256, 0, True, 1,
                             REF_VIDEO, REF_INFO, USE_IDLE_MODE, AUDIO_LENGTH, True, 
                             fps=20) """
    """ else:
        gr.Warning("不支持的方法：" + method) """
        #return None

    return (video, driven_vtt) if driven_vtt else video

def chat_response(system, message, history):
    # response = llm.generate(message)
    response, history = llm.chat(system, message, history)
    print(history)
    # 流式输出
    for i in range(len(response)):
        time.sleep(0.01)
        yield "", history[:-1] + [(message, response[:i+1])]
    return "", history

def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    llm.clear_history()
    return system, system, []

def clear_session():
    # clear history
    llm.clear_history()
    return '', []


GPT_SoVITS_ckpt = "GPT_SoVITS/pretrained_models"



iface = gr.Interface(
    fn=Talker_response_img,
    inputs=[gr.Textbox(label="输入文字/问题", lines=3, placeholder='请输入文本或问题，同时可以设置LLM模型。默认使用直接回复。')], 
    outputs=[gr.Video(label="数字人视频", format="mp4")],
    title="Image and Text to Video",  # Title of the app
    description="Upload an image and enter a caption to create a video that displays the image with the caption for 3 seconds."  # Description
)



def success_print(text):
    """输出绿色文本，表示成功信息。"""
    print(f"\033[1;32m{text}\033[0m")

def error_print(text):
    """输出红色文本，表示错误信息。"""
    print(f"\033[1;31m{text}\033[0m")
    
if __name__ == "__main__":
    # 初始化LLM类
    llm_class = LLM(mode='offline')
    llm = llm_class.init_model('直接回复 Direct Reply')
    success_print("默认不使用LLM模型，直接回复问题，同时减少显存占用！")

    # 尝试加载GPT-SoVITS模块


    # 尝试加载SadTalker模块
    try:
        from TFG import SadTalker
        talker = SadTalker(lazy_load=True)
        success_print("Success! SadTalker模块加载成功，默认使用SadTalker模型")
    except Exception as e:
        error_print(f"SadTalker 加载失败: {e}")
        error_print("如果使用SadTalker，请先下载SadTalker模型")

    # 尝试加载Whisper ASR模块
    try:
        from ASR import WhisperASR
        asr = WhisperASR('base')
        success_print("Success! WhisperASR模块加载成功，默认使用Whisper-base模型")
    except Exception as e:
        error_print(f"WhisperASR 加载失败: {e}")
        error_print("如果使用FunASR，请先下载WhisperASR模型并安装环境")

    # 检查GPU显存
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB
        if gpu_memory < 8:
            error_print("警告: 您的显卡显存小于8GB，不建议使用MuseTalk功能")

    # 尝试加载MuseTalk模块
    
    # 尝试加载EdgeTTS模块
    try:
        tts = edgetts
        if not tts.network:
            error_print("EdgeTTS模块加载失败，请检查网络连接")
    except Exception as e:
        error_print(f"EdgeTTS 加载失败: {e}")

    # Gradio UI的初始化和启动
    gr.close_all()
    """ demo_img = app_img()
    demo = gr.TabbedInterface(
        interface_list=[demo_img],
        tab_names=["个性化角色互动"],
        title="Linly-Talker WebUI"
    ) """
    """ demo.queue(max_size=4, default_concurrency_limit=2) """
    """ demo.launch(
        server_name=ip,  # 本地localhost:127.0.0.1 或 "0.0.0.0" 进行全局端口转发
        server_port=port,
        # ssl_certfile=ssl_certfile,  # SSL证书文件
        # ssl_keyfile=ssl_keyfile,  # SSL密钥文件
        # ssl_verify=False,
        share=True,
        debug=True,
    ) """
    import uvicorn

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://90bc-83-147-15-235.ngrok-free.app"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app1 = gr.mount_gradio_app(app, iface, path='/')
    uvicorn.run(app1, port=port, host=ip)
    """ iface.launch(
        server_name=ip,  # 本地localhost:127.0.0.1 或 "0.0.0.0" 进行全局端口转发
        server_port=port,
        # ssl_certfile=ssl_certfile,  # SSL证书文件
        # ssl_keyfile=ssl_keyfile,  # SSL密钥文件
        # ssl_verify=False,
        share=True,
        debug=True,
    ) """