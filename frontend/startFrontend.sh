docker build ./auntelisa -t auntelisa_frontend
docker run -d -p 3000:3000 --rm --name react_frontend_container auntelisa_frontend