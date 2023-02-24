# touch Dockerfile
# nano Dockerfile

FROM python:3.8-slim-buster

WORKDIR /app

COPY app/ .

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME ImageClassification

CMD ["python", "app.py"]
