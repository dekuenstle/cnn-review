FROM python:3.6

WORKDIR /app
ADD requirements.txt /app/
RUN pip3 install -r requirements.txt
ADD data/ /app/data/
ADD *.py entrypoint.sh /app/
ADD www/ /app/www/

EXPOSE 80
ENTRYPOINT ["./entrypoint.sh"]