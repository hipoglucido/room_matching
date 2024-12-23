# room_matching

# Commands

```
poetry shell
mlflow ui
black .
```
```
curl -X POST -H "Content-Type: application/json" -d '[{"A": "Meeting Room 1", "B": "Meeting Room One"}]' http://127.0.0.1:8080/predict
```

```commandline
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"referenceCatalog": ["Big room with balcony", "small suite"], "inputCatalog": ["Huge room along with a balcony", "nice balcony in big room", "luxury suite"]}' \
     http://127.0.0.1:8081/predict
```