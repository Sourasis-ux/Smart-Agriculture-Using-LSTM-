import requests

def generate_agricultural_report(moisture, temperature, humidity):
    prompt = f"""
    You are an automated Smart Agriculture AI monitoring an IoT sensor network. 
    Our Deep Learning LSTM model has just predicted the following sensor values for the upcoming hour:
    - Soil Moisture: {moisture:.2f}
    - Temperature: {temperature:.2f} °C
    - Humidity: {humidity:.2f} %
    
    Write a concise, 2-3 sentence diagnostic report for the farm manager. Briefly explain what these conditions mean for the crops and alert them if any action is needed.
    """
    
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"LLM Connection Failed: {str(e)}"

if __name__ == "__main__":
    print(generate_agricultural_report(25.5, 32.1, 80.0))