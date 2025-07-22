from pipeline_base import PodcastSummarizationPipeline

class SinglePassSummaryPipeline(PodcastSummarizationPipeline):
    def run(self, raw_text: str) -> dict:
        chunks = self.preprocess_and_chunk(raw_text)
        full_text = "\n\n".join(chunks)
        prompt = self.user_prompt_template.replace("{transcript}", full_text)
        summary = self.llm.generate(prompt=prompt, system_prompt=self.system_prompt)
        return {"summary": summary, "chunks_used": len(chunks)}