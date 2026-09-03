def build_golden(df_raw):
    return df_raw[
        [
            "query",
            "gold_doc_ids",
            "reference_answer",
            "task_type",
            "risk_level",
            "notes",
            "metadata_filter",
        ]
    ].copy()
