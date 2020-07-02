# Project Aunt Elisa

![logo](logo/granma.jpg)
![animation](logo/logo.mp4)


The project is about to have an web application with whom the user is able to interact.
Especially for lonely peoply and in connection to the actual corona crisis.
With our Product Aunt Eilza no one has to be anlone. Just talk with the chatbot and feel social proximity.
So let us fight social distancing in a positive way by social proximity to a chatbot.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
The Product is not ready to get used as a life system!

### Prerequisites

You need to install docker on your machine to run the project.
If you allready have docker installed, continue to next point.
Else visit [Docker](https://docs.docker.com/get-docker/) to install Docker software.

### Installing

This step by step series tells you how to get a development env running:

1. start docker software

2. go to Terminal and navigate to root of project folder, like this:


```
cd AuntElisa
```

3. now build docker-compose

```
docker-compose build
```

4. then run docker-compose

```
docker-compose up
```

5. the webapp is now available on:

```
localhost:3000
```


## Running the tests

**PLEASE USE CHROME BROWSER!**
Otherwise there is no guarantee that the application will work properly.

Access Landingpage at:

```
localhost:3000
```

There you see the welcome page with our logo (easter egg included)

By clicking on "START AUNT ELISA" the application gets usable.
You can return to Landingpage by clicking on the house icon top left.

Press "TALK" and speak to Elisa.
If you are ready click on "GET RESPONSE" to hear, what Elisa want's to response you.

Repeat the last two steps to enjoy a full conversatoin with the chatbot.



## Built With

* [React](https://reactjs.org) - for web application
* [Docker](https://www.docker.com) - application platform
* [Python](https://www.python.org) - used for backend and chatbot
* [Flask](https://pypi.org/project/Flask/) - used for connection to frontend
* [SpeechRecognition](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition) - speech to text
* [SpeechSynthesis](https://developer.mozilla.org/de/docs/Web/API/SpeechSynthesis) - text to speech


## Authors

* **Simon Scapan** - *Initial work* - [SimonScapan](https://github.com/SimonScapan)

* **Jannik Fischer** - *Contribution* - [Algebrator1997](https://github.com/Algebrator1997)

* **Johannes Deufel** - *Chatbot development* - [Johannes](https://github.com/Johannes998)

* **Simone Marx** - *Chatbot development* - [SimoneMarx](https://github.com/SimoneMarx)



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

