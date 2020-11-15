# Creatica Hackathon: "SeeFood"

## Created by Victoria Liu and Gloria Liu

## [See our website documenting our process](https://liuvictoria.github.io/creatica)


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

# 4. Open http://localhost:5000 on Safari and wait till the webpage is loaded. You may need to clear Safari cache if the app is not loading properly.
```

- Done! :tada:

:point_down: Screenshot:

<p align="center">
  <img src="https://user-images.githubusercontent.com/66798771/99161140-f3261b00-26bc-11eb-8d80-3db72b97a79f.png" height="420px" alt="">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/66798771/99161154-2668aa00-26bd-11eb-973e-e6b3deddfedb.png" height="420px" alt="">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/66798771/99161162-37192000-26bd-11eb-8ea7-2fe07dc95f40.png" height="420px" alt="">
</p>



#### License and original app

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[Original Flask application](https://github.com/mtobeiyf/keras-flask-deploy-webapp)


## References
[Griffin Chure's Reproducible Website](https://github.com/gchure/reproducible_website)

[Classification with Convolutional Neural Networks](https://towardsdatascience.com/building-the-hotdog-not-hotdog-classifier-from-hbos-silicon-valley-c0cb2317711f)

[InceptionNetV3](https://keras.io/api/applications/inceptionv3/)

[Deploying Keras Model with Flask](https://github.com/mtobeiyf/keras-flask-deploy-webapp)
