import os
import urllib.request
import ssl

output_dir = r"d:\learnthonproject\speech-emotion-analysis\REAL_HUMAN_SAMPLES"
os.makedirs(output_dir, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

base_url = "https://raw.githubusercontent.com/x4nth055/emotion-recognition-using-speech/master/data"
links = {
    "Angry": f"{base_url}/Actor_01/03-01-05-01-01-01-01.wav",
    "Happy": f"{base_url}/Actor_01/03-01-03-01-01-01-01.wav",
    "Neutral": f"{base_url}/Actor_01/03-01-01-01-01-01-01.wav",
    "Sad": f"{base_url}/Actor_01/03-01-04-01-01-01-01.wav"
}

for emotion, url in links.items():
    print(f"Downloading {emotion} from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as response:
            with open(os.path.join(output_dir, f"Human_{emotion}.wav"), 'wb') as f:
                f.write(response.read())
        print(f"✅ Saved {emotion} sample")
    except Exception as e:
        print(f"❌ Failed to download {emotion}: {e}")
