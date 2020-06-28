function Speak(input) {
    const synth = window.speechSynthesis;

    const speak = (text) => {
        var foo = new SpeechSynthesisUtterance(text);
        synth.speak(foo);
    };
    return speak(input);

}
export default Speak;






