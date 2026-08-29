import os
import io
import librosa
import soundfile as sf
from gtts import gTTS
import numpy as np

LANG_DATA = {
    "English": {
        "code": "en",
        "sentences": [
            "Spoken language identification is a key technology for speech recognition and translation.",
            "Welcome to the artificial intelligence audio classification demonstration.",
            "The quick brown fox jumps over the lazy dog in the sunny morning.",
            "We are developing modern deep learning architectures using recurrent neural networks.",
            "Machine learning models analyze acoustic features and mel frequency cepstral coefficients."
        ]
    },
    "Hindi": {
        "code": "hi",
        "sentences": [
            "नमस्ते, यह भाषा पहचान प्रणाली का एक वास्तविक भाषण नमूना है।",
            "भारत एक विशाल और सुंदर देश है जहाँ कई भाषाएँ बोली जाती हैं।",
            "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग तकनीक हमारे जीवन को बदल रही है।",
            "हम इस परियोजना में बोले जाने वाली भाषा को पहचानने का प्रयास कर रहे हैं।",
            "ध्वनि प्रसंस्करण और तंत्रिका नेटवर्क भाषण पहचान में बहुत उपयोगी हैं।"
        ]
    },
    "Mandarin": {
        "code": "zh-CN",
        "sentences": [
            "你好，这是一个用于语音语言识别的真实中文普通话样本。",
            "语言识别技术在人工智能和机器翻译中扮演着非常重要的角色。",
            "深度学习算法能够有效地分析声音信号中的特征模式。",
            "中国文化历史悠久，普通话是世界上使用人数最多的语言之一。",
            "我们正在使用循环神经网络来分类不同的语言发音。"
        ]
    },
    "Tamil": {
        "code": "ta",
        "sentences": [
            "வணக்கம், இது மொழி அடையாளம் காண்பதற்கான உண்மையான தமிழ் பேச்சு மாதிரி ஆகும்.",
            "தமிழ் மொழி மிகத் தொன்மையான மற்றும் வளமான இலக்கிய பாரம்பரியம் கொண்ட மொழி.",
            "செயற்கை நுண்ணறிவு தொழில்நுட்பம் மனித மொழிகளை மிகத் துல்லியமாக அடையாளம் காண்கிறது.",
            "இந்த திட்டத்தில் நாம் பல்வேறு இந்திய மொழிகளின் குரல் பதிவுகளை பகுப்பாய்வு செய்கிறோம்.",
            "ஒலி அலைவரிசைகள் மற்றும் பண்பேற்றங்கள் மூலமாக மொழி வகைப்படுத்தப்படுகிறது."
        ]
    },
    "Telugu": {
        "code": "te",
        "sentences": [
            "నమస్కారం, ఇది భాష గుర్తింపు కోసం నిజమైన తెలుగు మాట్లాడే నమూనా.",
            "తెలుగు భాష చాలా మధురమైనది మరియు దీనిని ఇటాలియన్ ఆఫ్ ది ఈస్ట్ అని పిలుస్తారు.",
            "కృత్రిమ మేధస్సు మరియు డీప్ లెర్నింగ్ ద్వారా ధ్వని సంకేతాలను విశ్లేషిస్తున్నాము.",
            "ఈ ప్రాజెక్ట్ లో మేము వివిధ భారతీయ భాషల ప్రసంగాలను గుర్తిస్తున్నాము.",
            "భాష గుర్తింపు వ్యవస్థ వివిధ సాంకేతిక రంగాలలో విస్తృతంగా ఉపయోగించబడుతుంది."
        ]
    },
    "Kannada": {
        "code": "kn",
        "sentences": [
            "ನಮಸ್ಕಾರ, ಇದು ಭಾಷಾ ಗುರುತಿಸುವಿಕೆಗಾಗಿ ನೈಜ ಕನ್ನಡ ಭಾಷಣದ ಧ್ವನಿ ಮಾದರಿಯಾಗಿದೆ.",
            "ಕನ್ನಡ ನಾಡು ಮತ್ತು ನುಡಿ ಅತ್ಯಂತ ಶ್ರೀಮಂತ ಸಾಂಸ್ಕೃತಿಕ ಇತಿಹಾಸವನ್ನು ಹೊಂದಿದೆ.",
            "ಆರ್ಟಿಫಿಶಿಯಲ್ ಇಂಟೆಲಿಜೆನ್ಸ್ ತಂತ್ರಜ್ಞಾನವು ಧ್ವನಿ ಆವರ್ತನಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ.",
            "ಈ ಯೋಜನೆಯಲ್ಲಿ ನಾವು ವಿವಿಧ ಪ್ರಾದೇಶಿಕ ಭಾಷೆಗಳನ್ನು ವರ್ಗೀಕರಿಸುತ್ತಿದ್ದೇವೆ.",
            "ಭಾಷಾ ಗುರುತಿಸುವಿಕೆ ಮಾದರಿಗಳು ನರಮಂಡಲ ಜಾಲಗಳ ಮೂಲಕ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತವೆ."
        ]
    },
    "Malayalam": {
        "code": "ml",
        "sentences": [
            "നമസ്കാരം, ഇത് ഭാഷ തിരിച്ചറിയുന്നതിനുള്ള യഥാർത്ഥ മലയാള സംഭാഷണ സാമ്പിളാണ്.",
            "കേരളത്തിലെ മനോഹരമായ ഭാഷയാണ് മലയാളം, ഇതിന് സമ്പന്നമായ സാഹിത്യ പാരമ്പര്യമുണ്ട്.",
            "ശബ്ദ തരംഗങ്ങൾ വിശകലനം ചെയ്തുകൊണ്ട് കമ്പ്യൂട്ടറുകൾക്ക് ഭാഷ തിരിച്ചറിയാൻ സാധിക്കും.",
            "ഈ പ്രോജക്റ്റിൽ ആർട്ടിഫിഷ്യൽ ഇന്റലിജൻസ് ഉപയോഗിച്ച് ഞങ്ങൾ ഭാഷ കണ്ടെത്തുന്നു.",
            "ആധുനിക ഡീപ് ലേണിംഗ് മോഡലുകൾ സംഭാഷണ പ്രക്രിയയിൽ വലിയ മുന്നേറ്റമുണ്ടാക്കുന്നു."
        ]
    },
    "Bengali": {
        "code": "bn",
        "sentences": [
            "নমস্কার, এটি ভাষা শনাক্তকরণের জন্য একটি প্রকৃত বাংলা কথ্য অডিও নমুনা।",
            "বাংলা অত্যন্ত মিষ্টি এবং সমৃদ্ধ একটি ভাষা যা বিশ্বজুড়ে বহু মানুষ বলে থাকেন।",
            "কৃত্রিম বুদ্ধিমত্তা মানুষের কণ্ঠস্বর বিশ্লেষণ করে সঠিক ভাষা চিহ্নিত করতে পারে।",
            "আমরা ডিপ লার্নিং এবং নিউরাল নেটওয়ার্কের মাধ্যমে ভাষা শনাক্তকরণ করছি।",
            "শব্দ তরঙ্গ এবং মেল ফ্রিকোয়েন্সি বৈশিষ্ট্যের সাহায্যে ভাষার পার্থক্য বোঝা যায়।"
        ]
    },
    "Marathi": {
        "code": "mr",
        "sentences": [
            "नमस्कार, हा भाषा ओळखण्यासाठी एक खरा मराठी बोलण्याचा ऑडिओ नमुना आहे.",
            "मराठी भाषा ही महाराष्ट्राची समृद्ध आणि ऐतिहासिक भाषा आहे.",
            "आर्टिफिशियल इंटेलिजन्स तंत्रज्ञान आवाजाच्या लहरींचे विश्लेषण करून भाषा ओळखते.",
            "या प्रकल्पामध्ये आम्ही विविध भारतीय भाषांचे वर्गीकरण करत आहोत.",
            "डीप लर्निंग मॉडेल्स भाषण ओळख प्रणालीमध्ये अत्यंत प्रभावी ठरतात."
        ]
    },
    "Gujarati": {
        "code": "gu",
        "sentences": [
            "નમસ્તે, આ ભાષા ઓળખ માટેનો એક અસલી ગુજરાતી બોલવાનો નમૂનો છે.",
            "ગુજરાતી ભાષા ભારતના પશ્ચિમ ભાગમાં બોલાતી એક ખૂબ જ લોકપ્રિય ભાષા છે.",
            "આર્ટિફિશિયલ ઇન્ટેલિજન્સ અવાજના આધારે યોગ્ય ભાષાની ઓળખ કરી શકે છે.",
            "આ પ્રોજેક્ટમાં અમે વિવિધ પ્રાદેશિક ભાષાઓના અવાજનું વિશ્લેષણ કરીએ છીએ.",
            "કમ્પ્યુટર મોડલ્સ શબ્દો અને ધ્વનિ તરંગોને સરળતાથી ઓળખી શકે છે."
        ]
    },
    "Punjabi": {
        "code": "pa",
        "sentences": [
            "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ, ਇਹ ਭਾਸ਼ਾ ਪਛਾਣ ਲਈ ਇੱਕ ਅਸਲੀ ਪੰਜਾਬੀ ਬੋਲਣ ਵਾਲਾ ਨਮੂਨਾ ਹੈ।",
            "ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਪੰਜਾਬ ਦੇ ਲੋਕਾਂ ਦੀ ਬਹੁਤ ਹੀ ਮਿੱਠੀ ਅਤੇ ਜੋਸ਼ੀਲੀ ਬੋਲੀ ਹੈ।",
            "ਆਰਟੀਫਿਸ਼ੀਅਲ ਇੰਟੈਲੀਜੈਂਸ ਆਵਾਜ਼ ਦੇ ਤਰੰਗਾਂ ਦੀ ਪਛਾਣ ਕਰਕੇ ਭਾਸ਼ਾ ਦੱਸ ਸਕਦੀ ਹੈ।",
            "ਅਸੀਂ ਇਸ ਪ੍ਰੋਜੈਕਟ ਵਿੱਚ ਵੱਖ-ਵੱਖ ਭਾਰਤੀ ਭਾਸ਼ਾਵਾਂ ਨੂੰ ਵਰਗੀਕ੍ਰਿਤ ਕਰ ਰਹੇ ਹਾਂ।",
            "ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਮਾਡਲ ਆਵਾਜ਼ ਦੇ ਫੀਚਰਾਂ ਨੂੰ ਸਮਝ ਕੇ ਸਹੀ ਨਤੀਜਾ ਦਿੰਦੇ ਹਨ।"
        ]
    },
    "French": {
        "code": "fr",
        "sentences": [
            "Bonjour, ceci est un enregistrement vocal authentique pour l'identification de la langue parlée.",
            "La langue française possède une riche tradition littéraire et culturelle dans le monde entier.",
            "L'intelligence artificielle permet de classifier les langues humaines avec une grande précision.",
            "Dans ce projet, nous analysons les caractéristiques acoustiques et fréquentielles de la voix humaine.",
            "Les réseaux de neurones récurrents apprennent à distinguer les accents et les phonèmes linguistiques."
        ]
    },
    "German": {
        "code": "de",
        "sentences": [
            "Guten Tag, dies ist eine authentische Sprachaufnahme zur automatischen Erkennung gesprochener Sprachen.",
            "Die deutsche Sprache zeichnet sich durch präzise grammatikalische Strukturen und reiche Wortbildungen aus.",
            "Künstliche Intelligenz und Deep Learning revolutionieren die moderne Sprachverarbeitung und Mustererkennung.",
            "In diesem Projekt analysieren wir akustische Merkmale und Frequenzspektren von menschlichen Sprachsignalen.",
            "Neuronale Netze ermöglichen eine zuverlässige Klassifikation verschiedener Weltsprachen und Dialekte."
        ]
    },
    "Spanish": {
        "code": "es",
        "sentences": [
            "Hola, esta es una muestra de audio auténtica para la identificación automática del idioma hablado.",
            "El español es uno de los idiomas más hablados en todo el mundo con una gran riqueza cultural.",
            "La inteligencia artificial y el aprendizaje profundo transforman los sistemas de reconocimiento de voz.",
            "En este proyecto analizamos las características acústicas y los coeficientes cepstrales de la voz humana.",
            "Las redes neuronales recurrentes permiten clasificar con alta precisión diferentes idiomas del mundo."
        ]
    }
}

def generate_speech_audio(target_dir="./sample_audio", dataset_dir="./real_speech_dataset"):
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    target_sr = 16000
    
    print("Generating authentic human speech audio clips via gTTS...")
    for lang_name, data in LANG_DATA.items():
        lang_code = data["code"]
        lang_folder = os.path.join(dataset_dir, lang_name.lower())
        os.makedirs(lang_folder, exist_ok=True)
        
        for idx, sentence in enumerate(data["sentences"]):
            tts = gTTS(text=sentence, lang=lang_code, slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            # Load mp3 bytes into 16kHz float32 audio
            audio, sr = librosa.load(mp3_fp, sr=target_sr, mono=True)
            
            # Pad / loop audio to 10.0 seconds (160,000 samples)
            desired_len = int(10.0 * target_sr)
            if len(audio) < desired_len:
                repeats = int(np.ceil(desired_len / len(audio)))
                audio = np.tile(audio, repeats)[:desired_len]
            else:
                audio = audio[:desired_len]
                
            # Normalize volume
            audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.95
            
            # Save to dataset folder
            save_path = os.path.join(lang_folder, f"speech_{idx:02d}.wav")
            sf.write(save_path, audio, target_sr)
            
            # Save the first sentence as the primary sample in ./sample_audio/
            if idx == 0:
                sample_path = os.path.join(target_dir, f"sample_{lang_name.lower()}.wav")
                sf.write(sample_path, audio, target_sr)
                print(f"  [OK] Generated authentic spoken audio for: {lang_name} ({lang_code})")

if __name__ == "__main__":
    generate_speech_audio()
