### Problemy:
- testowanie trzech podejść do generowania uszkodzeń (cupy, skimage, numpy) - numpy wypadł najlepiej
- próba zapisu do h5, hugging face tego nie obsługuje
- ostatecznie zapisywanie datasetu lokalnie za pomocą datasets (hugging face)
- dataset waży 13GB, ponieważ obrazy z wygenerowanym uszkodzeniem nie zostały skompresowane
- ostatecznie użycie PIL do wygenerowania uszkodzeń i kompresowanie do JPEG na wyjściu

### Pytania:
- czym są kolumny embeddings i resnet w datasecie i czy będą nam potrzebne
- dzielenie na zbiór treningowy testowy i walidacyjny ze stratyfikacją na jakie klasy (gatunek, styl?)
- kompresja do JPEG sprawia, że obraz z uszkodzeniem różni się od oryginału nie tylko w miejscu uszkodzenia. (rozwiązanie: ~~w metodzie ```JpegImageFile.save()``` parametr ```subsampling='keep'```~~ zapisać jako PNG)