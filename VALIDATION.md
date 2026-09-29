# Provera paketa

Lokalno na Windows-u:

- `tests/test_selfism.py`: 8 testova prolazi.
- Provereni novi API/UI, dozvoljeni profili, zaključavanje tokom drugih instalacija/generisanja, uvezani workflowi i trigger, ograničeni log i prekid/timeout instalacionog procesa.
- JavaScript sintaksa originalnog i dodatog interfejsa proverena preko `node --check`.
- Modeli i custom-node izvori su navedeni u `catalog/selfism.json`. Civitai SHA256 se pribavlja pre preuzimanja pomoću korisnikovog tokena.

Build i preostale provere:

- Docker build je prosao na GitHub Actions: https://github.com/digitrgrs-eng/selfism-dashboard/actions/runs/36507212645 . Objavljeni digest: `sha256:8903c8b5a7251ed7b53f20d0594b44daf342752dbc31f2744f11a1fb39971ce2`. Ovo ne potvrđuje GPU izvrsavanje.
- Instalacija svih Python zavisnosti u Linux GPU image-u i stvarno generisanje Simple/AIO.
- Privatni Civitai download sa korisnikovim tokenom, RapidCache prijava i ubrzanje dodatih modela.
- Kompletna originalna test kolekcija: postoje neusaglašena očekivanja broja originalnih workflowa i testovi koji očekuju Linux/ComfyUI okruženje. Paket se ne predstavlja kao potpuno integraciono testiran.

Pre stvarnog korišćenja: uspešan image build, novi probni GPU pod, log `Qwen35ChatHandler: True` i `GPU offload: True`, restart ComfyUI, zatim po jedno uspešno Simple i AIO generisanje. Proveriti i postojeći originalni workflow i RapidCache prijavu.
