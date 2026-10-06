# Selfora / Selfism dashboard za digitrgrs-eng

Build koristi gotov `10sorllabs/comfyui-workflow-launcher:2.0`, zakljucan na linux/amd64 manifest `sha256:d01908958aa33cc9117b478d81845d14ed53c135f173e3db3eafe14e780e8846`. Dodaje dashboard pomocu `COPY --link`, bez ponovne instalacije CUDA/Python/dlib baze. Jedan `RUN` (`docker/bake_comfyui.sh`) ugradjuje ComfyUI v0.38.1 (git checkout, `origin` = Comfy-Org/ComfyUI) u `/opt/comfyui-baked` i instalira njegove zavisnosti u system site-packages uz zasticene torch/torchvision/torchaudio/numpy/transformers/pillow/opencv; zato novi pod ne instalira nista pri prvom startu, a `selfism_boot` samo potvrdi da je ComfyUI vec ispravan. Posto taj `RUN` mora da raspakuje ~11 GiB base rootfs, workflow vec oslobadja disk (oko 114 GiB slobodno) pre builda. Postojeci `/workspace` volumeni zadrzavaju svoj stari ComfyUI dok ih `selfism_boot` pri startu ne prebaci na v0.38.1 (sada brzo, jer su zavisnosti vec u image-u). RunPod i dalje preuzima kompletan image. Autorov eventualno ugradjeni HF credential se ne koristi (`HF_TOKEN_FILE=/dev/null`); koristi svoj `HF_TOKEN`.

Pripremljen iz korisnikovog 10sorLabs dashboard arhiva. Originalni Workflows, Custom models, Custom nodes i RapidCache interfejs ostaju prisutni. Dodat je Selfora / Selfism panel sa instalacijama Simple, AIO, dodatnih modela i Qwen/CUDA popravkom, uz prikaz izlaza instalacije.

## Trenutni status

Aktuelni buildovi i njihovi rezultati su u [GitHub Actions](https://github.com/digitrgrs-eng/selfism-dashboard/actions/workflows/build-image.yml). Svaki uspesan build objavljuje `ghcr.io/digitrgrs-eng/selfism-dashboard:latest` i nepromenljiv tag sa punim commit SHA. Novi Carousel GPU test na RunPod-u jos nije izvrsen. Pogledaj VALIDATION.md.

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
- `SELFISM_AUTO_COMFY_UPDATE=1` (podrazumevano) — pri svakom startu poda, PRE pokretanja ComfyUI-a, `launcher/selfism_boot.py` pokreće `selfism_int8.py`: ako ComfyUI nema `int8_tensorwise` ili je `comfy-kitchen` < 0.2.16, prebacuje ga na v0.38.1 (uz `PIP_CONSTRAINT` zaštitu torch/numpy/transformers i rollback). Kada je već ažurno, traje nekoliko sekundi. Greška nikad ne blokira start; log je `/workspace/selfism-boot.log`. `0` isključuje, `force` zanemaruje limit ponovnih pokušaja (nakon 2 uzastopna neuspeha za isti tag boot ne pokušava ponovo). `SELFISM_AUTO_COMFY_UPDATE_TIMEOUT` (sekunde, podrazumevano 1800).
- `CIVITAI_TOKEN` — lični Civitai API token za preuzimanje Selfora modela.
- `HF_TOKEN` — tvoj Hugging Face read token, ako koristiš originalne workflowe sa modelima koji zahtevaju pristup. Prihvati njihove uslove na Hugging Face-u.

Tokeni se unose u RunPod; ne ugrađuju se u Docker image. RapidCache koristi postojeću prijavu i postojeće uslove naloga. Pre instalacije prijavi se u RapidCache. Simple, AIO, dodatni i pojedinacni modeli proveravaju svez katalog tog naloga. Ubrzani HTTPS izvor koristi se samo kada SHA256 i poznata velicina odgovaraju originalu. Zadrzava se odredisni folder workflowa i provera integriteta. Bez podudaranja ili ako katalog nije dostupan koristi se originalni link. Proverava se katalog koji API izlozi nalogu, ne pretrazuje se ceo privatni RapidCache storage. Log prikazuje izbor izvora bez potpisanih URL-ova. Selfora i dalje zahteva Civitai metapodatke/token da bi se utvrdio originalni SHA256. Ako R2 transfer ne uspe, proverava se RapidCache i zatim originalni link. Ako RapidCache transfer ne uspe, pokusava se original. Otkazivanje i greske diska ne pokrecu novi izvor. SHA256 i odredisna putanja ostaju isti.

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

## Privatni R2

Dodaj ove environment promenljive u RunPod template i pod:

- SELFISM_R2_ENDPOINT=https://58b03040f369a1ae324ceac0da1dacb6.r2.cloudflarestorage.com
- SELFISM_R2_BUCKET=selfism-models
- SELFISM_R2_ACCESS_KEY_ID: privatni Read only Access Key ID
- SELFISM_R2_SECRET_ACCESS_KEY: privatni Read only Secret Access Key

Kljuceve ne unositi u GitHub. Instalater najpre proverava privatni bucket, zatim RapidCache katalog, zatim originalne izvore. R2 objekti moraju imati metadata sha256 koju postavlja upload skripta; mora se poklopiti sa pouzdanim katalogom i velicinom. Fajlovi bez pouzdane SHA256 u katalogu ostaju na izvornim linkovima. Civitai token i dalje je potreban za originalne Selfora metapodatke. Potpisani linkovi traju 24 sata. Redosled R2 -> RapidCache -> original vazi i pri mreznoj gresci zapocetog transfera. Ako svi dostupni izvori zakazu, instalacija se prijavljuje kao neuspesna. Brzina nije garantovana i treba je izmeriti na pod-u.


## 10sorLabs Reference + Depth
The Selfora / Selfism page includes a separate Reference + Depth installer. It uses Krea 2 Turbo FP8 regardless of the Selfora precision dropdown and installs the reference captioner, Depth control, Artfat Resolution and existing optional face/upscale dependencies. Source preference is the shared verified private R2 -> RapidCache -> original URL resolver. RapidCache requires an exact SHA256 match; availability is not assumed.

Depth Anything weights are stored under `models/controlnet_aux` and linked into the annotator checkpoint directory after node installation, avoiding an untracked first-run download. The normal `--all` R2 upload includes these and all newly cataloged models. Millie remains a private, separately supplied LoRA; an existing `millie_000002750.safetensors` or `millie.safetensors` is enabled when the installed workflow is first saved. Existing saved workflows are preserved.

After installing, use the RunPod Jupyter terminal to upload available catalog files:

```bash
PYTHONPATH=/opt/r2deps python3 /opt/10sorlabs/scripts/upload_models_r2.py --all
```

The tool prompts for bucket-scoped upload credentials without saving them, verifies local files, and skips already matching objects. Runtime download credentials remain read-only. No upload is triggered by installing a workflow.


## AIO Qwen Carousel (2026-09-30)

U Selfora / Selfism dodato je **Download & install AIO Qwen Carousel**. Paket koristi Selfora FP8 za prvu sliku i Qwen-Image-Edit-2511 FP8 mixed za edit, nezavisno od dropdown izbora za Simple/AIO. Nije potrebno prethodno kliknuti Simple, AIO ili Reference + Depth.

Dugme instalira Selfora osnovu, postojeci LLM i mmproj, Depth i Depth Anything, originalni upscaler, Qwen edit/encoder/VAE, person-seg model i potrebne custom nodove. Pomocni `ComfyUI-AIO-Carousel` dolazi iz ovog repozitorijuma; `color-matcher` i `ultralytics` se instaliraju u stvarni ComfyUI `.venv-cu128`, uz zasticene verzije osnovnih paketa. Raniji pomocni nodovi se kopiraju u `user/carousel_backups` pre zamene. Privatna Millie LoRA se ne preuzima: ako vec postoji pod jednim od podrzanih imena, povezuje se pri prvom cuvanju workflowa.

Sva cetiri Qwen dodatka imaju iste putanje, velicine i SHA256 kao `upload_qwen_carousel_r2.py`. Postojeci lokalni modeli se proveravaju i preskacu. Runtime R2 kredencijali ostaju **Read only**; upload kredencijale nije potrebno dodavati u dashboard. Selfora metadata i dalje zahteva `CIVITAI_TOKEN`.

Instalacija cuva `Selfism_AIO_m1lli3_CAROUSEL_QWEN_2511_v1.json` u `ComfyUI/user/default/workflows/Selfism`, a postojecu korisnicku verziju ne prepisuje. Posle uspesne instalacije ComfyUI se automatski restartuje. JSON se moze preuzeti i direktno iz nove kartice.

Prvi test: `Use approved first photo`, jedna ukljucena referenca 02, reference 03-09 OFF, radna rezolucija 1 MP. Ovo je isti carousel paket koji je pripremljen za GPU test, bez novog generativnog koraka. Testovi instalatera ne potvrduju vizuelni kvalitet niti VRAM zahteve.

Promena GitHub repozitorijuma ili objavljivanje novog Docker image-a ne menja automatski vec pokrenuti pod. Novi pod treba da koristi novi image tag iz uspesnog build-a. Postojeci uploadovani modeli u R2 ostaju dostupni.

## MiniMax H3 Simply Advanced
Kartica **MiniMax H3 Simply Advanced** (dugme **Instaliraj**) na stranici Selfora / Selfism instalira samo podrazumevano aktivne fajlove workflowa „Simply Advanced“ (altoiddealer v1.4, `selfism_workflows/minimax_h3_simply_advanced.json`): Ref2VA INT8 ConvRot (20,97 GB), Qwen3-VL 32B encoder INT4 ConvRot (14,95 GB), video VAE INT8 ConvRot (2,81 GB), audio VAE FP32 (0,61 GB), latent upscaler FP16 (0,69 GB) i Gemma-4 E4B INT8 ConvRot za LLM korak prompta (8,09 GB), ukupno oko 48,1 GB, plus nodovi KJNodes, rgthree, Easy-Use, Logic, Spectrum MiniMax H3, Latent Upscaler, comfyui-various, VideoHelperSuite, essential-er, LLM Text Processor i MiniMax H3 RefMod (plus Python paket `soundfile` koji comfyui-various traži pri učitavanju; ComfyUI-Logic je zakačen na verziju koja ima čvor `Bool`). Workflow se čuva nepromenjen (isti bajtovi) kao `user/default/workflows/Selfism/Simply_Advanced_MiniMax_H3_v1.4.json` i ne prepisuje se ako već postoji; može se preuzeti i sa `/api/selfism/workflow/minimax`.

Opcioni fajlovi su u katalogu kao pojedinačni modeli: FL2VA INT8, NVFP4 encoder (radi i bez Blackwell GPU-a), FP16 video VAE, Gemma-4 E2B, turbo LoRA i TaoMate LoRA (Civitai, traži `CIVITAI_TOKEN`). Svi su preslikani u privatni R2 bucket `selfism-models` pod ključem `<destination bez models/>` sa `sha256` metapodatkom, pa instalacija prvo koristi R2, a zatim Hugging Face/Civitai. Postojeća kartica „MiniMax H3“ (`catalog/workflows.json`) nije menjana i i dalje preuzima sa Hugging Facea. Napomena: ako SageAttention nije instaliran (launcher ga gradi samo za Hopper/Blackwell), u podgrafu „Backend Attention“ izaberi „pytorch attention“. Nije testirano na GPU podu.

Kartica **MiniMax H3 R2V Turbo (Hearmeman)** instalira workflow HearmemanAI „MiniMax H3 R2V“ (`selfism_workflows/minimax_h3_r2v_turbo_hearmeman.json`, čuva se kao `user/default/workflows/Selfism/MiniMax_H3_R2V_Turbo_Hearmeman.json`): Ref2VA INT8 (34 GB), Qwen3-VL 32B encoder INT8 (27 GB), video VAE FP16, audio VAE FP32, Ref2V turbo 4-step LoRA plus rgthree, KJNodes, VideoHelperSuite i ComfyUI-MiniMaxRefPack (Hearmeman24, v0.3.5). Workflow je isti kao original osim što je u UNETLoaderu uklonjen prefiks `diffusion_models/` (fajl je instaliran u `models/diffusion_models/`) i ispravljen metapodatak o preuzimanju. Opcioni HM* LoRA-ovi se ne instaliraju; red „hmmotion“ je isključen. Latent preview u image-u je none (CLI `comfyui_args.txt`, Settings i Manager), a ModelPreviewOverrideKJ ima tiny_vae=none. Čvor „MiniMax References Manager“ po defaultu piše prompt preko OpenRouter-a, zato u pod okruženje dodaj `OPENROUTER_API_KEY` (ili `LLM_KEY`) ili u čvoru izaberi `prompt_provider = none`.

Kartica **MiniMax H3 R2V Swap Low-VRAM (Hearmeman)** (profil `minimax_r2v_swap_lowvram`) instalira `selfism_workflows/minimax_h3_r2v_swap_lowvram.json` (čuva se kao `user/default/workflows/Selfism/MiniMax_H3_R2V_Swap_LowVRAM_Hearmeman.json`, postojeći fajl se ne prepisuje). Napravljen je od Hearmeman R2V Turbo workflowa (kartica `minimax_r2v` i njen JSON nisu menjani): čvor MiniMax References Manager ne prima IMAGE ulaze, pa su `LoadImage` „Picture 1 (original woman from video)“, `LoadImage` „Picture 2 (Millie / new person)“ i `VHS_LoadVideo` „Video 1 (source reel)“ (custom_width 640, custom_height 0, force_rate 24, format H3, `frame_load_cap` vezan za izračunatu dužinu) povezani direktno na core `MiniMaxH3ReferenceToVideo` (`ref_image_0`, `ref_image_1`, `ref_video_0`, a zvuk reela na `ref_video_audio_0`). Prompt za zamenu je u čvoru „Prompt“ (PrimitiveStringMultiline), bez OpenRoutera. Modeli: Ref2VA INT8 pruned (`minimax_h3_ref2va_pruned_int8_convrot.safetensors`), Qwen3-VL 32B NVFP4 AWQ encoder, video VAE FP16, audio VAE FP32 i Ref2V turbo 4-step LoRA (4 koraka, isti sampler/scheduler), plus rgthree i VideoHelperSuite; oko 44,4 GB, sve iz R2 (`selfism-models`) sa Hugging Face rezervom. Podrazumevano 9:16, 0,35 MP (448x800), 5 s (124 frejma), `ref_image_size = match`; nema ModelPreviewOverrideKJ/taeh3 ni hmmotion LoRA reda. Ako reel nema audio traku, obriši vezu `audio` sa čvora Video 1 (VHS inače javlja grešku pri čitanju zvuka). Nije testirano na GPU podu.

Kartica **MiniMax H3 R2V Swap High-Res (Hearmeman)** (profil `minimax_r2v_swap_highres`) instalira `selfism_workflows/minimax_h3_r2v_swap_highres.json` (čuva se kao `user/default/workflows/Selfism/MiniMax_H3_R2V_Swap_HighRes_Hearmeman.json`, postojeći fajl se ne prepisuje). Klon Low-VRAM swap kartice sa VRAM uštedama za 0,75–0,98 MP (~11 s) na ~97 GB GPU: isti modeli (~44,4 GB) plus KJNodes (`MiniMaxChunkFeedForward` chunks=4/seq_threshold=2048 i `MiniMaxLowVRAMAttention` head_chunks=4 posle Turbo LoRA). Video 1 na širini 512; podrazumevano 9:16, 0,98 MP (768x1344), 5 s (za ceo reel 11 s); `ref_image_size = max`; jači identity-swap prompt. Instalacija dodaje `--reserve-vram 8` u workspace `comfyui_args.txt` ako fajl postoji (restart ComfyUI). Opcioni Sage attention nije uključen. Kartice Low-VRAM i R2V Turbo nisu menjane. Nije testirano na GPU podu.

## Kompletan workflow (Selfora Full)
Na stranici Selfora / Selfism postoji kartica **Kompletan workflow** sa dugmetom **Instaliraj sve**. Instalira samo modele i custom nodove koje Recreate workflow `selfism_workflows/full.json` stvarno koristi: Selfora v2.1 (INT8 je podrazumevan, FP8 kao alternativa; BF16 nije u ovom paketu), Qwen3-VL encoder i Qwen VAE, Artfat LLM Prompter (`RVN-Q4_K_M-multilingual-mtp.gguf` + `mmproj-Qwen3.8-27B-Q8_0.gguf`), upscalere i skin detail, detektore (lice, oči, kosa), SAM, Depth Control LoRA + Depth Anything V2, Millie LoRA (`millie_000002750.safetensors`, sa Hugging Facea `daneheh/millie`), Power LoRA slajdere i nodove. Identity Edit, OpenPose LoRA, DWPose, detektor ruku i FableVibes LLM više nisu deo paketa (Identity Edit i OpenPose mogu se instalirati pojedinačno). Procenjeno oko 41 GB (INT8) / 40,3 GB (FP8) slobodnog prostora.

Instalacija kopira osam LLM preseta u `ComfyUI/models/LLM/prompts/` (prepisuje ih): četiri `Recreate_{SFW,NSFW}_{prefix,noprefix}.txt` (v2, identity-safe: telo/lice/koža dolaze iz Millie LoRA, LLM opisuje samo pozu, kadar, scenu, garderobu i frizuru) i četiri `Recreate_FirstFrame_{SFW,NSFW}_{prefix,noprefix}.txt` (za prvi frame reela: precizna poza tela i glave, zamrznut still, kadar 9:16; slika se posle koristi kao Picture 1 u MiniMax H3). Posle instalacije restartuj ComfyUI i izaberi preset u dropdown-u prompter node-a. Preset fajl na podu ima prednost nad tekstom koji je upakovan u node, pa ovo drži pod usklađen sa workflowom.

Izvori su isti kao kod ostalih paketa: privatni R2 -> RapidCache -> originalni link, sa istom SHA256 proverom. Svi fajlovi iz ovog paketa imaju SHA256 i veličinu u `catalog/selfism.json`. Workflow se čuva kao `user/default/workflows/Selfism/Selfism_FULL_<format>_recreate_v1.json`; postojeći fajl se ne prepisuje. JSON se može preuzeti i direktno sa kartice. Civitai LoRA fajlovi traže `CIVITAI_TOKEN`. Putanje LoRA/GGUF u workflowu su svedene na imena fajlova u osnovnim folderima. Nije testirano na GPU podu.
