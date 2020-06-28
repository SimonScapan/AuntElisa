function Hear() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    let paragraph = document.createElement('p');

    const dictate = () => {
        recognition.start();
        recognition.onresult = (event) => {
            const speechToText = event.results[0][0].transcript;
            paragraph.textContent = speechToText;
        }
    }
    dictate();
    return(paragraph);
}
export default Hear;

