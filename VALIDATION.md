# Provera paketa

Lokalno na Windows-u:

- `tests/test_selfism.py`: 8 testova prolazi.
- Provereni novi API/UI, dozvoljeni profili, zaključavanje tokom drugih instalacija/generisanja, uvezani workflowi i trigger, ograničeni log i prekid/timeout instalacionog procesa.
- JavaScript sintaksa originalnog i dodatog interfejsa proverena preko `node --check`.
- Modeli i custom-node izvori su navedeni u `catalog/selfism.json`. Civitai SHA256 se pribavlja pre preuzimanja pomoću korisnikovog tokena.

Nije potvrđeno:

- Docker build: Docker nije dostupan na lokalnoj mašini.
- Instalacija svih Python zavisnosti u Linux GPU image-u i stvarno generisanje Simple/AIO.
- Privatni Civitai download sa korisnikovim tokenom, RapidCache prijava i ubrzanje dodatih modela.
- Kompletna originalna test kolekcija: postoje neusaglašena očekivanja broja originalnih workflowa i testovi koji očekuju Linux/ComfyUI okruženje. Paket se ne predstavlja kao potpuno integraciono testiran.

Pre stvarnog korišćenja: uspešan image build, novi probni GPU pod, log `Qwen35ChatHandler: True` i `GPU offload: True`, restart ComfyUI, zatim po jedno uspešno Simple i AIO generisanje. Proveriti i postojeći originalni workflow i RapidCache prijavu.
