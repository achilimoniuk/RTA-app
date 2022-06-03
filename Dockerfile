FROM python:3.8

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT [ "python" ]
CMD [ "app-1.py" ]
