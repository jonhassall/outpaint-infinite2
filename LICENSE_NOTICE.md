# Upstream licensing notice

This starter project does not redistribute the Krea 2 base-model weights, the outpaint LoRA weights, or the custom pipeline code. They are downloaded at runtime from Hugging Face.

You must review and comply with:

- the **Krea 2 Community License** and Krea Acceptable Use Policy for the base model and LoRA weights;
- the upstream `PIPELINE_LICENSE` and `NOTICE` files for the custom pipeline/helper implementation;
- any content-moderation, disclosure, and deployment obligations that apply to your use case.

The custom pipeline revision is pinned by default because it is loaded with `trust_remote_code=True`. Review the upstream code before changing that revision.
