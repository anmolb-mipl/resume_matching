# Add/replace these parts in your existing api.py.
# 1) Extend SearchRequest with this field:
#    contextual: bool = Field(
#        True,
#        description="Use Gemini to evaluate shortlisted candidates in whole-resume context."
#    )
#
# 2) Replace the existing search endpoint with:
#
# @app.post("/search")
# def search(req: SearchRequest):
#     output = rank_candidates(
#         req.jd_text, req.top_n, req.min_score,
#         use_rerank=req.rerank, use_context=req.contextual
#     )
#     if output["error"]:
#         code = 409 if "index is empty" in output["error"] else 422
#         raise HTTPException(status_code=code, detail=output["error"])
#     return output
#
# If you want Gemini to be used by default, keep contextual=True. Set it to false
# in the request to run local retrieval/reranking only.
