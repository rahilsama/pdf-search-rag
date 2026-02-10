from core.rag_pipeline import run_rag


def main() -> None:
    """
    Simple CLI entry point for querying the local RAG system.
    """
    query = input("Enter your question: ")

    result = run_rag(query)

    answer = result["answer"]
    sources = result["sources"]
    latency_seconds = result["latency_seconds"]

    print("\nAI Answer:\n")
    print(answer)

    if sources:
        print("\nSources:")
        for idx, src in enumerate(sources, start=1):
            source_str = src.get("source") or "Unknown source"
            chunk_index = src.get("chunk_index")
            print(f"{idx}. {source_str} (chunk {chunk_index})")

    print(f"\nLatency: {latency_seconds:.2f} seconds")


if __name__ == "__main__":
    main()