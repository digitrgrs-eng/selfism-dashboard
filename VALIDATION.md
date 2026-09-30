# Provera paketa

Lokalno na Windows-u:

- `tests/test_selfism.py`: 17 testova prolazi.
- Provereni novi API/UI, dozvoljeni profili, zaključavanje tokom drugih instalacija/generisanja, uvezani workflowi i trigger, ograničeni log i prekid/timeout instalacionog procesa.
- JavaScript sintaksa originalnog i dodatog interfejsa proverena preko `node --check`.
- Modeli i custom-node izvori su navedeni u `catalog/selfism.json`. Civitai SHA256 se pribavlja pre preuzimanja pomoću korisnikovog tokena.

Build i preostale provere:

- Docker build je prosao na GitHub Actions: https://github.com/digitrgrs-eng/selfism-dashboard/actions/runs/36507212645 . Objavljeni digest: `sha256:8903c8b5a7251ed7b53f20d0594b44daf342752dbc31f2744f11a1fb39971ce2`. Ovo ne potvrđuje GPU izvrsavanje.
- Instalacija svih Python zavisnosti u Linux GPU image-u i stvarno generisanje Simple/AIO.
- Privatni Civitai download sa korisnikovim tokenom, RapidCache prijava i ubrzanje dodatih modela.
- Kompletna originalna test kolekcija: postoje neusaglašena očekivanja broja originalnih workflowa i testovi koji očekuju Linux/ComfyUI okruženje. Paket se ne predstavlja kao potpuno integraciono testiran.

Pre stvarnog korišćenja: uspešan image build, novi probni GPU pod, log `Qwen35ChatHandler: True` i `GPU offload: True`, restart ComfyUI, zatim po jedno uspešno Simple i AIO generisanje. Proveriti i postojeći originalni workflow i RapidCache prijavu.

RapidCache testovi: isto ime sa razlicitim SHA256 ne podudara se; razlicita velicina, HTTP, neubrzani izvor i pogresan auth ostaju na originalu. Ispravno podudaranje cuva lokalni folder, ukljucuje SHA256 proveru, ne loguje potpisani URL. Nedostupan API vraca originalne izvore. Prijavljeni RapidCache nalog na pravom pod-u jos nije testiran.


Reference + Depth addition: 24 focused tests pass (Selfism and private R2), including Turbo selection, static loader model coverage, depth checkpoint link, UI/API exposure and R2 inventory. No live GPU generation, private R2 upload or deployment on an active pod was performed.


## AIO Qwen Carousel — 2026-09-30

- Novi carousel UI/API profil ima kompletnu pokrivenost statickih loadera katalogom; Qwen UNET ostaje Qwen, a Selfora FP8 ostaje prva grana nezavisno od dropdown-a.
- Sva cetiri nova modela imaju iste SHA256, velicine i kljuceve kao izvrsena R2 seed skripta. SHA256 upscalera izracunat je iz zvanicnog GitHub release fajla (29,663,426 bajtova).
- Provereni R2 -> RapidCache -> origin transfer fallback, stvarni downloader preko simuliranog HTTP 403 i ispravnih bajtova sa origin-a, verifikacija, otkazivanje, greske diska i uklanjanje neispravnih .part fajlova bez diranja finalnih modela.
- Provereni instalacija pomocnih zavisnosti u prosledjeni ComfyUI Python, backup prethodnih pomocnih nodova, cuvanje korisnickih izmena workflowa i privatne LoRA-e.
- Python sintaksa i JavaScript sintaksa prolaze. Carousel JSON je identican prethodno dostavljenom AIO_QWEN_CAROUSEL_v1 paketu.
- Fokusirana kolekcija sada ima 35 testova. CI izvrsava tu kolekciju pre Docker build-a.
- Sira lokalna kolekcija pri proveri: 221 prolazi, 6 pada. Svih sest padova ponovljeno je na neizmenjenom prethodnom HEAD-u 5024932 u zasebnom privremenom direktorijumu: pet testova ocekuje 5 originalnih workflowa umesto 4, jedan ocekuje 5 sidebar stavki umesto 6. Nisu novonastale regresije. Dve naknadno dodate Carousel provere takodje prolaze.
- Nije izvrsena stvarna GPU generacija, instalacija na korisnikovom aktivnom podu niti pristup njegovom privatnom R2 bucketu. Uspeh Docker build-a je odvojena provera u GitHub Actions.
