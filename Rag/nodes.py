from Rag.Pinecone.store import vector_store
from Rag.funtion import decompose_to_sentences,filter_chain,llm,doc_eval_chain
from langchain_core.documents import Document
from typing import List
from Rag.state import llm_cls
import time
from dotenv import load_dotenv

load_dotenv()



def retrieve(state:llm_cls):
    start = time.time()
    print("Enter the Retrieved")
    question = state['messages'][-1].content
    docs = vector_store.similarity_search(query=question,
                                   k=5)
    print(f"Retrieved after {time.time()-start}")
    print(f"Retrieved {len(docs)} docs")
    return {"docs":docs}
    
def eval_doc(state: llm_cls):
    UPPER_TH = 0.7
    LOWER_TH = 0.3

    q = state["messages"][-1].content
    scores: List[float] = []
    reasons: List[str] = []
    good: List[Document] = []

    for d in state["docs"]:
        out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
        print(f"chunk = {d.page_content} \n Doc Eval: score={out.score},\n reason={out.reason}")
        scores.append(out.score)
        reasons.append(out.reason)

        # 5) for CORRECT case we will refine only docs with score > LOWER_TH
        if out.score > LOWER_TH:
            good.append(d)

    # 2) CORRECT if at least one doc > UPPER_TH
    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
        }

    # 3) INCORRECT if all docs < LOWER_TH
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        why = "No chunk was sufficient."
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}. {why}",
        }
    why = "Mixed relevance signals."
    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}. {why}",
    }


def refine(state:llm_cls):
    start = time.time()
    q = state['messages'][-1].content
    context="\n\n".join(d.page_content for d in state["docs"]).strip()

    strips = decompose_to_sentences(context)

    kept: List[str] = []
    for s in strips:
        if filter_chain.invoke({"question": q, "sentence": s}).keep:
            kept.append(s)

    refined_context = "\n".join(kept).strip()
    print(f"Refined after {time.time()-start}")
    return {
        "strips":strips,
        "keep_strips":kept,
        "refined":refined_context
    }

def generate(state:llm_cls):
    start = time.time()
    querry = state['messages']
    context = state['docs']
    print(context)
    prompt=f"""You are a context-based question answering assistant.

            You must answer the user ONLY using the information from the CONTEXT extracted from the uploaded book.

            Strict Rules:
            1. If the answer is clearly present in the CONTEXT, you may rephrase it into a short, natural sentence,
            but you MUST NOT add any new information.
            2. If the answer is NOT present, respond EXACTLY with:
            Not found in the given context.
            3. Do NOT use any external knowledge.
            4. Do NOT guess, assume, or invent facts.
            5. Do NOT add details that are not stated in the CONTEXT.
            6. Do NOT mention the word "context" in your answer.
            7. If only part of the answer is found, answer only with what is available.

            CONTEXT:
            {context}

            USER QUESTION:
            {querry}

            ANSWER:
            """
    res = llm.invoke(prompt)
    print(f"Generate after : {time.time()-start}")
    return {
        'messages':res.content
    }
