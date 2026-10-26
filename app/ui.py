"""Interface web Gradio : Interroger · Bibliothèque · Réglages."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gradio as gr

from . import answer, db, ingest, llm, store
from .config import DEFAULT_SYSTEM_PROMPT, Settings, cfg, load_settings, save_settings
from .extract import SUPPORTED
from .retrieve import reranker_available

STATUS_LABELS = {"ok": "✅ indexé", "ocr_needed": "⚠️ OCR requis"}


# ------------------------------------------------------------------ état / santé


def health_markdown() -> str:
    s = load_settings()
    lines = []
    try:
        models = llm.list_models(s)
        lines.append(f"- **LM Studio** : ✅ joignable ({len(models)} modèle(s) disponible(s))")
    except llm.LLMError as exc:
        lines.append(f"- **LM Studio** : ❌ {exc}")
    ok, info = store.ping()
    st = db.stats()
    if ok:
        lines.append(f"- **Qdrant** : ✅ {store.count()} vecteurs")
    else:
        lines.append(f"- **Qdrant** : ❌ injoignable ({info})")
    lines.append(f"- **Bibliothèque** : {st['documents']} document(s), {st['chunks']} passages")
    lines.append(f"- **Modèle de chat** : `{s.llm_model or 'non choisi'}`")
    indexed = db.get_meta("embed_model")
    lines.append(f"- **Modèle d'embedding** : `{s.embed_model or 'non choisi'}`"
                 + (f" (index construit avec `{indexed}`)" if indexed else ""))
    if indexed and s.embed_model and indexed != s.embed_model:
        lines.append("- ⚠️ **Le modèle d'embedding a changé depuis l'indexation** : lancez « Réindexer tout » "
                     "dans la Bibliothèque, sinon les questions seront refusées.")
    if s.use_reranker:
        lines.append(f"- **Reranker** : {'✅ actif' if reranker_available() else '⚠️ demandé mais indisponible (voir README)'}")
    return "\n".join(lines)


def banner_markdown() -> str:
    s = load_settings()
    st = db.stats()
    if not s.llm_model or not s.embed_model:
        return "👉 **Première utilisation** : ouvrez l'onglet **⚙️ Réglages**, testez la connexion à LM Studio et choisissez vos deux modèles."
    if st["documents"] == 0:
        return "📚 La bibliothèque est vide : ajoutez des documents dans l'onglet **Bibliothèque**."
    return f"📚 {st['documents']} document(s) · {st['chunks']} passages indexés"


# ------------------------------------------------------------------ interroger


def _plain(content) -> str:
    """Gradio peut renvoyer le contenu d'un message sous forme de blocs [{'type': 'text', 'text': ...}]."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def respond(question: str, history: list[dict], spaces: list[str], top_k: int):
    question = (question or "").strip()
    history = history or []
    if not question:
        yield history, gr.skip(), ""
        return
    past = [{"role": m["role"], "content": _plain(m["content"])} for m in history]
    history = past + [{"role": "user", "content": question},
                      {"role": "assistant", "content": "🔎 Recherche dans les documents…"}]
    yield history, "_Recherche en cours…_", ""
    try:
        prep = answer.prepare(question, past, spaces or None, int(top_k))
    except Exception as exc:  # noqa: BLE001
        history[-1]["content"] = f"⚠️ {exc}"
        yield history, "_Aucune source._", ""
        return
    sources = answer.format_sources(prep.passages, prep.standalone, prep.question, prep.retrieval_seconds)
    history[-1]["content"] = "✍️ Rédaction de la réponse…"
    yield history, sources, ""
    text = ""
    try:
        for piece in answer.stream_answer(prep):
            text += piece
            history[-1]["content"] = text
            yield history, sources, ""
        if not text.strip():
            history[-1]["content"] = "_Le modèle n'a renvoyé aucun texte._"
    except Exception as exc:  # noqa: BLE001
        history[-1]["content"] = (text + "\n\n" if text else "") + f"⚠️ {exc}"
    yield history, sources, ""


def clear_chat():
    return [], "_Aucune source._", ""


# ------------------------------------------------------------------ bibliothèque


def _doc_rows() -> list[list]:
    return [
        [d["id"], d["title"], d["filename"], d["collection"], d["pages"] or "", d["n_chunks"],
         STATUS_LABELS.get(d["status"], d["status"]), d["added_at"]]
        for d in db.list_documents()
    ]


def _doc_choices() -> list[tuple[str, int]]:
    return [(f"{d['collection']} · {d['filename']}", d["id"]) for d in db.list_documents()]


def library_refresh():
    """Met à jour tableau, sélecteurs de documents et listes d'espaces."""
    spaces = db.list_collections()
    return (
        _doc_rows(),
        gr.update(choices=_doc_choices(), value=None),
        gr.update(choices=spaces),
        gr.update(choices=spaces, value=[]),
        banner_markdown(),
    )


def do_upload(files, collection, progress=gr.Progress()):
    collection = (collection or "Général").strip() or "Général"
    if not files:
        yield "Aucun fichier sélectionné.", *library_refresh()
        return
    if not ingest.ingest_lock.acquire(blocking=False):
        yield "⏳ Une indexation est déjà en cours, patientez avant d'en lancer une autre.", *library_refresh()
        return
    log: list[str] = []
    try:
        paths = [Path(f if isinstance(f, str) else f.name) for f in files]
        for i, path in enumerate(paths):
            def report(frac: float, msg: str, i=i) -> None:
                progress((i + frac) / len(paths), desc=msg)

            try:
                r = ingest.ingest_file(path, collection, filename=path.name, progress=report, keep_copy=True)
                log.append(r.line())
            except Exception as exc:  # noqa: BLE001
                log.append(f"❌ {path.name} — {exc}")
            yield "\n".join(log), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    finally:
        ingest.ingest_lock.release()
    yield "\n".join(log) + "\n\nTerminé.", *library_refresh()


def do_scan_inbox(collection, progress=gr.Progress()):
    collection = (collection or "Général").strip() or "Général"
    if not ingest.ingest_lock.acquire(blocking=False):
        yield "⏳ Une indexation est déjà en cours.", *library_refresh()
        return
    try:
        results = ingest.scan_inbox(collection, lambda frac, msg: progress(frac, desc=msg))
    finally:
        ingest.ingest_lock.release()
    text = "\n".join(r.line() for r in results) or f"Le dossier inbox est vide ({cfg.inbox_dir})."
    yield text + "\n\nTerminé.", *library_refresh()


def do_delete(doc_id, confirm):
    if doc_id is None:
        return "Choisissez d'abord un document.", *library_refresh()
    if not confirm:
        return "Cochez « Confirmer la suppression » pour supprimer.", *library_refresh()
    doc = db.get_document(int(doc_id))
    if doc is None:
        return "Document introuvable.", *library_refresh()
    ingest.remove_document(int(doc_id))
    return f"🗑️ « {doc['filename']} » supprimé de l'index.", *library_refresh()


def do_move(doc_id, new_collection):
    if doc_id is None:
        return "Choisissez d'abord un document.", *library_refresh()
    if not (new_collection or "").strip():
        return "Indiquez le nom du nouvel espace.", *library_refresh()
    ingest.move_document(int(doc_id), new_collection)
    return f"📂 Document déplacé vers « {new_collection.strip()} ».", *library_refresh()


def do_reindex(progress=gr.Progress()):
    if not ingest.ingest_lock.acquire(blocking=False):
        yield "⏳ Une indexation est déjà en cours.", *library_refresh()
        return
    try:
        n = ingest.reindex_all(lambda frac, msg: progress(frac, desc=msg))
        msg = f"✅ {n} passages réindexés avec « {load_settings().embed_model} »."
    except Exception as exc:  # noqa: BLE001
        msg = f"❌ {exc}"
    finally:
        ingest.ingest_lock.release()
    yield msg, *library_refresh()


# ------------------------------------------------------------------ réglages


def _model_update(models: list[str], current: str, prefer_embedding: bool):
    emb = [m for m in models if llm.looks_like_embedding_model(m)]
    chat = [m for m in models if not llm.looks_like_embedding_model(m)]
    ordered = (emb + chat) if prefer_embedding else (chat + emb)
    value = current if current else (ordered[0] if ordered else None)
    return gr.update(choices=ordered, value=value)


def do_test(url, key, cur_llm, cur_embed):
    probe = replace(load_settings(), lmstudio_base_url=url.strip(), api_key=key)
    try:
        models = llm.list_models(probe)
    except llm.LLMError as exc:
        return f"❌ {exc}", gr.skip(), gr.skip()
    if not models:
        return ("⚠️ Connexion OK mais aucun modèle n'est disponible. Téléchargez-en dans LM Studio "
                "(un modèle de chat et un modèle d'embedding)."), gr.skip(), gr.skip()
    return (f"✅ Connexion OK — {len(models)} modèle(s) trouvé(s). Choisissez ci-dessous, puis enregistrez.",
            _model_update(models, cur_llm, False), _model_update(models, cur_embed, True))


def do_save(url, key, llm_model, embed_model, temperature, top_k, max_ctx, use_rr, condense,
            qprefix, dprefix, system_prompt):
    old = load_settings()
    new = Settings(
        lmstudio_base_url=url.strip(), api_key=key, llm_model=llm_model or "", embed_model=embed_model or "",
        embed_query_prefix=qprefix or "", embed_doc_prefix=dprefix or "", temperature=float(temperature),
        top_k=int(top_k), max_context_chars=int(max_ctx), use_reranker=bool(use_rr),
        condense_history=bool(condense), system_prompt=(system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT,
    )
    save_settings(new)
    msg = "✅ Réglages enregistrés."
    indexed = db.get_meta("embed_model")
    if indexed and new.embed_model and indexed != new.embed_model:
        msg += (f"\n\n⚠️ Le modèle d'embedding (`{new.embed_model}`) diffère de celui de l'index (`{indexed}`). "
                "Allez dans **Bibliothèque → Réindexer tout** avant de poser des questions.")
    if new.use_reranker and not reranker_available():
        msg += ("\n\n⚠️ Reranker demandé mais indisponible : l'image a-t-elle été construite avec "
                "`WITH_RERANKER=1` ? (voir README)")
    return msg, health_markdown(), banner_markdown(), gr.update(value=new.top_k)


def reset_prompt():
    return DEFAULT_SYSTEM_PROMPT


# ------------------------------------------------------------------ application


def build_app() -> gr.Blocks:
    cfg.ensure_dirs()
    s = load_settings()
    spaces0 = db.list_collections()

    with gr.Blocks(title="Bibliothèque RAG") as demo:
        gr.Markdown("# 📚 Bibliothèque — interrogez vos documents\n"
                    "Tout reste sur votre machine : LM Studio, Qdrant et cette interface.")
        banner = gr.Markdown(banner_markdown())

        with gr.Tabs():
            # ---------------------------------------------------------- Interroger
            with gr.Tab("💬 Interroger"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chat = gr.Chatbot(height=540, placeholder="Posez une question sur vos documents…")
                        with gr.Row():
                            msg = gr.Textbox(show_label=False, scale=8, lines=1, autofocus=True,
                                             placeholder="Votre question (Entrée pour envoyer)")
                            send = gr.Button("Envoyer", variant="primary", scale=1)
                        clear = gr.Button("Nouvelle conversation", size="sm")
                    with gr.Column(scale=2):
                        spaces = gr.CheckboxGroup(
                            label="Espaces interrogés", choices=spaces0, value=[],
                            info="Aucun coché = toute la bibliothèque.")
                        topk = gr.Slider(2, 15, value=s.top_k, step=1, label="Nombre de passages retrouvés")
                        gr.Markdown("### Sources")
                        sources = gr.Markdown("_Aucune source._")

            # ---------------------------------------------------------- Bibliothèque
            with gr.Tab("📚 Bibliothèque"):
                gr.Markdown(
                    "Formats : " + ", ".join(sorted(SUPPORTED)) + ". "
                    "Nommez vos fichiers `Auteur_Titre_Année.pdf` : le nom apparaît dans les citations. "
                    "Un PDF déjà indexé et inchangé est ignoré ; un fichier modifié remplace l'ancienne version."
                )
                with gr.Row():
                    with gr.Column():
                        files = gr.File(label="Déposer des documents", file_count="multiple",
                                        file_types=sorted(SUPPORTED))
                        space_in = gr.Dropdown(label="Espace (thème)", choices=spaces0, value="Général",
                                               allow_custom_value=True,
                                               info="Choisissez un espace existant ou tapez-en un nouveau.")
                        with gr.Row():
                            up_btn = gr.Button("Indexer ces fichiers", variant="primary")
                            inbox_btn = gr.Button("Indexer le dossier inbox")
                        gr.Markdown(f"_Dossier inbox : `{cfg.inbox_dir}` — chaque sous-dossier devient un espace._")
                    log = gr.Textbox(label="Journal d'indexation", lines=14, interactive=False)

                table = gr.Dataframe(
                    value=_doc_rows(),
                    headers=["ID", "Titre", "Fichier", "Espace", "Pages", "Passages", "Statut", "Ajouté"],
                    interactive=False, wrap=True, label="Documents indexés")

                with gr.Accordion("Gérer un document", open=False):
                    doc_pick = gr.Dropdown(label="Document", choices=_doc_choices(), value=None)
                    with gr.Row():
                        with gr.Column():
                            new_space = gr.Dropdown(label="Nouvel espace", choices=spaces0, allow_custom_value=True)
                            move_btn = gr.Button("Déplacer")
                        with gr.Column():
                            confirm = gr.Checkbox(label="Confirmer la suppression")
                            del_btn = gr.Button("🗑️ Supprimer de l'index", variant="stop")
                with gr.Accordion("Maintenance", open=False):
                    gr.Markdown("**Réindexer tout** recalcule tous les vecteurs à partir du texte déjà extrait. "
                                "À faire après un changement de modèle d'embedding (compter quelques minutes).")
                    reindex_btn = gr.Button("Réindexer tout")

            # ---------------------------------------------------------- Réglages
            with gr.Tab("⚙️ Réglages"):
                gr.Markdown("### Connexion à LM Studio\nDémarrez le serveur local dans LM Studio (onglet *Developer*), "
                            "chargez un modèle de chat et un modèle d'embedding, puis testez la connexion.")
                with gr.Row():
                    url = gr.Textbox(label="URL de l'API", value=s.lmstudio_base_url, scale=3)
                    key = gr.Textbox(label="Clé API (facultatif)", value=s.api_key, type="password", scale=1)
                test_btn = gr.Button("Tester la connexion et lister les modèles")
                test_out = gr.Markdown()
                with gr.Row():
                    llm_dd = gr.Dropdown(label="Modèle de langage (chat)", allow_custom_value=True,
                                         choices=[s.llm_model] if s.llm_model else [], value=s.llm_model or None)
                    emb_dd = gr.Dropdown(label="Modèle d'embedding", allow_custom_value=True,
                                         choices=[s.embed_model] if s.embed_model else [], value=s.embed_model or None,
                                         info="Un seul modèle pour tout l'index : en changer impose de réindexer.")
                with gr.Row():
                    temperature = gr.Slider(0, 1, value=s.temperature, step=0.05, label="Température")
                    top_k_set = gr.Slider(2, 15, value=s.top_k, step=1, label="Passages par défaut")
                    max_ctx = gr.Number(value=s.max_context_chars, precision=0, label="Contexte max (caractères)",
                                        info="~4 caractères par token ; à adapter à la fenêtre de contexte du modèle.")
                with gr.Row():
                    use_rr = gr.Checkbox(value=s.use_reranker, label="Reranker (meilleure précision, plus lent)")
                    condense = gr.Checkbox(value=s.condense_history, label="Reformuler les questions de suivi")
                with gr.Accordion("Avancé", open=False):
                    qprefix = gr.Textbox(label="Préfixe des requêtes (embedding)", value=s.embed_query_prefix,
                                         info="Vide pour BGE-M3. Pour multilingual-e5 : « query: ».")
                    dprefix = gr.Textbox(label="Préfixe des documents (embedding)", value=s.embed_doc_prefix,
                                         info="Vide pour BGE-M3. Pour multilingual-e5 : « passage: ».")
                    sys_prompt = gr.Textbox(label="Consignes données au modèle", value=s.system_prompt, lines=10)
                    reset_btn = gr.Button("Rétablir les consignes par défaut", size="sm")
                save_btn = gr.Button("Enregistrer", variant="primary")
                save_out = gr.Markdown()
                gr.Markdown("### État du système")
                health = gr.Markdown(health_markdown())
                refresh_health = gr.Button("Actualiser", size="sm")

        # ------------------------------------------------------------ événements
        chat_io = dict(inputs=[msg, chat, spaces, topk], outputs=[chat, sources, msg])
        msg.submit(respond, **chat_io)
        send.click(respond, **chat_io)
        clear.click(clear_chat, outputs=[chat, sources, msg])

        lib_outputs = [table, doc_pick, space_in, spaces, banner]
        up_btn.click(do_upload, [files, space_in], [log, *lib_outputs])
        inbox_btn.click(do_scan_inbox, [space_in], [log, *lib_outputs])
        move_btn.click(do_move, [doc_pick, new_space], [log, *lib_outputs])
        del_btn.click(do_delete, [doc_pick, confirm], [log, *lib_outputs])
        reindex_btn.click(do_reindex, None, [log, *lib_outputs])

        test_btn.click(do_test, [url, key, llm_dd, emb_dd], [test_out, llm_dd, emb_dd])
        save_btn.click(do_save, [url, key, llm_dd, emb_dd, temperature, top_k_set, max_ctx, use_rr, condense,
                                 qprefix, dprefix, sys_prompt], [save_out, health, banner, topk])
        reset_btn.click(reset_prompt, outputs=sys_prompt)
        refresh_health.click(health_markdown, outputs=health)

        demo.load(lambda: (banner_markdown(), health_markdown(), gr.update(choices=db.list_collections())),
                  outputs=[banner, health, spaces])

    demo.queue(default_concurrency_limit=4)
    return demo


def main() -> None:
    demo = build_app()
    auth = (cfg.app_user, cfg.app_password) if cfg.app_user and cfg.app_password else None
    demo.launch(server_name=cfg.host, server_port=cfg.port, auth=auth, theme=gr.themes.Soft(), show_error=True)


if __name__ == "__main__":
    main()
