# Start here

This folder contains the complete rebuilt project. Extract the ZIP before running code.

1. Read [README.md](README.md) for setup, execution and the experiment.
2. Review [the two-page write-up](deliverables/writeup_two_pages.pdf). A [one-page alternative](deliverables/writeup_one_page.pdf) covers the conflicting length instructions.
3. Open [the editable PowerPoint](deliverables/presentation.pptx), or [its PDF companion](deliverables/presentation.pdf). [Speaker notes](docs/presentation_notes.md) explain each slide.
4. Follow [the demo and viva guide](docs/demo_and_viva.md) and inspect the executed [analysis notebook](notebooks/01_analysis.ipynb).
5. Use [the assignment checklist](docs/deliverables_checklist.md) and [GitHub submission guide](docs/github_submission.md). Push this folder's contents, including dotfiles, into the private project repository.

## Personalize before submission

Names and SRNs are included in the report, presentation and README. To edit them later, update `docs/team_details.json`, then regenerate the PDFs:

```text
python -m pip install -r requirements-docs.txt
python scripts/build_writeups.py
```

Edit the title slide directly in PowerPoint if needed. Fill in the real contribution record in `docs/team_contributions.md`. Repository: https://github.com/Atharva-Ashish-Vyas/corporate-bankruptcy-prediction. The team will change repository visibility to private in GitHub Settings before formal submission, then invite faculty and the assigned TA using their course-provided accounts.

## Verification completed

- Full six-model experiment executed and saved.
- Six automated tests passed, covering separation of data, prediction schema, threshold selection, inference consistency, reported counts and app uploads.
- All six notebook code cells executed successfully.
- All 14 slide previews and all three write-up pages visually reviewed.
- PowerPoint package, layout, editable tables and six embedded chart workbooks validated.

The slide PDF is a fixed visual copy; use the PowerPoint for editing. Native Microsoft PowerPoint rendering and a GitHub-hosted Actions run were not tested in this environment.
