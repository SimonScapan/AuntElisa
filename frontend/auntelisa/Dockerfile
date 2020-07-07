# base image for the container
FROM node:12.2.0-alpine

# set WORKDIR
WORKDIR /app

# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/:$PATH

# install and cache app dependencies
# COPY package.json /app/package.json
#COPY ./package.json /app/package.json
COPY . .
RUN npm install --silent
RUN npm install react-scripts@3.0.1 -g --silent
Run npm install @material-ui/core --no-audit

Copy ./ /app
# start app
CMD ["npm","start"]
