# Creatica Hackathon: "SeeFood"

## Created by Victoria Liu and Gloria Liu

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[Original Flask application](https://github.com/mtobeiyf/keras-flask-deploy-webapp)

[See our website documenting our process](liuvictoria.github.io/creatica)

# To Use
- Clone this repo 
- Install requirements
- Run the script
- Go to http://localhost:5000
- Done! :tada:

:point_down: Screenshot:

<p align="center">
  <img src="https://user-images.githubusercontent.com/5097752/71063354-8caa1d00-213a-11ea-86eb-879238887c1f.png" height="420px" alt="">
</p>



## Run with Docker

With **[Docker](https://www.docker.com)**, you can quickly build and run the entire application in minutes :whale:

```shell
# 1. First, clone the repo
$ git clone https://github.com/mtobeiyf/keras-flask-deploy-webapp.git
$ cd keras-flask-deploy-webapp

# 2. Build Docker image
$ docker build -t hotdgg .

# 3. Run!
$ docker run -it --rm -p 5000:5000 hotdgg
```

Open http://localhost:5000 on Safari and wait till the webpage is loaded. You may need to clear Safari cache if the app is not loading properly.



## References
[Griffin Chure's Reproducible Website](https://github.com/gchure/reproducible_website)
[Classification with Convolutional Neural Networks](https://towardsdatascience.com/building-the-hotdog-not-hotdog-classifier-from-hbos-silicon-valley-c0cb2317711f)
[InceptionNetV3](https://keras.io/api/applications/inceptionv3/)
[Deploying Keras Model with Flask](https://github.com/mtobeiyf/keras-flask-deploy-webapp)
