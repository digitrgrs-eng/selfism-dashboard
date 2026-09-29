# Selfora / Selfism dashboard za digitrgrs-eng

Build koristi gotov `10sorllabs/comfyui-workflow-launcher:2.0`, zakljucan na linux/amd64 manifest `sha256:d01908958aa33cc9117b478d81845d14ed53c135f173e3db3eafe14e780e8846`. Dodaje samo dashboard pomocu `COPY --link`, bez ponovne instalacije CUDA/Python/dlib baze. Prenosi oko 5,46 GiB kompresovanih postojecih slojeva izmedju registara, bez potrebe da ih raspakuje za RUN komande. Zbog promenjenog nacina izrade sada proverava najmanje 15 GiB slobodnog prostora. RunPod i dalje preuzima kompletan image. Autorov eventualno ugradjeni HF credential se ne koristi (`HF_TOKEN_FILE=/dev/null`); koristi svoj `HF_TOKEN`.

Pripremljen iz korisnikovog 10sorLabs dashboard arhiva. Originalni Workflows, Custom models, Custom nodes i RapidCache interfejs ostaju prisutni. Dodat je Selfora / Selfism panel sa instalacijama Simple, AIO, dodatnih modela i Qwen/CUDA popravkom, uz prikaz izlaza instalacije.

## Trenutni status

Paket je lokalno pripremljen. Nije objavljen na GitHub-u, Docker image nije izgrađen i nije izvršen GPU test na RunPod-u. GitHub korisničko ime samo po sebi ne daje pristup nalogu. Pogledaj VALIDATION.md.

## 1. GitHub

1. Prijavi se kao `digitrgrs-eng` i napravi **privatan** repozitorijum `selfism-dashboard` na https://github.com/new .
2. Raspakuj dostavljeni ZIP. U repozitorijum postavi **sadržaj** direktorijuma `selfism-dashboard`, a ne ZIP fajl. `Dockerfile` mora biti u korenu repozitorijuma.
3. Proveri da je postavljen i skriveni direktorijum `.github/workflows/build-image.yml`. Ako ga Windows ne prikazuje, uključi prikaz skrivenih stavki.
4. U GitHub-u otvori **Actions → Build RunPod image → Run workflow**. Pokretanje je ručno, ne na svaki upload.
5. Sačekaj uspešan build. Ciljna adresa je `ghcr.io/digitrgrs-eng/selfism-dashboard:latest`. Sama adresa nije dokaz da image postoji; potreban je zeleni rezultat build-a.

GitHub Actions na privatnom repozitorijumu ima besplatnu kvotu. Ovaj image je veliki: standardni runner možda nema dovoljno diska ili vremena. Ako build ne uspe zbog resursa, sačuvaj log; nemoj uključivati plaćeni runner bez provere troška. Modeli se ne uključuju u image.

Privatan image zahteva RunPod Registry credentials za `ghcr.io`: GitHub username i token sa `read:packages`. Token unesi direktno u RunPod, ne u repo ili poruku. Javno objavljivanje izvornih 10sorLabs fajlova/image-a nije urađeno ovim paketom; pre promene vidljivosti proveriti uslove autora.

## 2. RunPod template — tek posle uspešnog build-a

Napravi novi template koristeći podešavanja postojećeg 10sorLabs template-a:

| Polje | Vrednost |
| --- | --- |
| Container image | `ghcr.io/digitrgrs-eng/selfism-dashboard:latest` (ili immutable SHA tag iz Actions rezultata) |
| Container start command | Ostavi prazno |
| HTTP ports | `3000,8188,8888` |
| TCP port | `22`, ako koristiš SSH |
| Volume mount | `/workspace` |
| Volume Disk | Kao do sada, prema ukupnoj veličini izabranih modela; ne Network Volume |
| Registry credentials | GitHub nalog sa pristupom privatnom paketu |

Environment variables:

- `LAUNCHER_AUTO_UPDATE=0` — obavezno, da originalni auto-update ne zameni dodatak.
- `SELFISM_AUTO_REPAIR=1` — nakon dostupnosti ComfyUI-a pokreće Qwen/CUDA popravku.
- `CIVITAI_TOKEN` — lični Civitai API token za preuzimanje Selfora modela.
- `HF_TOKEN` — tvoj Hugging Face read token, ako koristiš originalne workflowe sa modelima koji zahtevaju pristup. Prihvati njihove uslove na Hugging Face-u.

Tokeni se unose u RunPod; ne ugrađuju se u Docker image. RapidCache koristi postojeću prijavu i postojeće uslove naloga. Novi Selfism modeli koriste navedene originalne izvore; nije potvrđeno da ih RapidCache ubrzava.

## 3. Svaki novi pod

1. Otvori dashboard na portu 3000. Sačekaj da ComfyUI bude spreman i automatska popravka završena.
2. U **Selfora / Selfism** izaberi FP8, INT8 ili BF16, zatim instalaciju Simple ili AIO. Preuzimanje ide direktno na pod.
3. Ako se pojavi greška, pročitaj status/log. Instalacija se neće prikazati kao uspešna kada zavisnosti zakažu. Dugme za Qwen/CUDA popravku možeš ponovo pokrenuti kada ComfyUI ne generiše.
4. Nakon instalacije/popravke koristi postojeće **Restart ComfyUI** dugme da proces učita nove biblioteke i nodove.
5. Workflow se čuva u `ComfyUI/user/default/workflows/Selfism`. Učitaj svoju referentnu sliku.
6. Svoju privatnu `millie_000002750.safetensors` LoRA-u preuzmi zasebno kroz Custom models u `loras`, pa uključi njen red u Power LoRA Loader-u. Uključeni workflowi koriste `m1lli3` i zadržavaju Power LoRA Loader.

Već izmenjen workflow istog imena ne prepisuje se automatski. Dodatni AIO modeli su zaseban izbor. Nisu sve opcione grane podrazumevano uključene.

Popravka cilja Linux x86_64, Python 3.12 i CUDA 12.8 iz ove baze. Proverava Qwen35ChatHandler i stvaran GPU offload; ne rešava nedostatak VRAM-a. Test generisanja na ciljnom GPU-u ostaje obavezan pre nego što template smatraš proverenim.

## Izvor

Osnova: https://github.com/10sorlabs/AI1-Model-Grabber , prateći fajlovi sa revizije `8a94a3d6c0b8ca0164aea15f8e37439156ce021c`. Postojeći autorski sadržaj i oznake zadržani su. Ovo nije zvanično 10sorLabs izdanje.
