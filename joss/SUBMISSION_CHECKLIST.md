# JOSS submission checklist

Based on all author- and reviewer-facing pages of the JOSS documentation
(https://joss.readthedocs.io, checked 5 November 2026): Submitting, Paper
format, Example paper, Policies, Reviewing for JOSS, Review criteria,
Review checklist, Editorial guide, and EditorialBot. `[x]` = done,
`[ ]` = open. Delete this file before the branch is submitted.

## 1. Pre-review gates (failing one means desk rejection; falling short on several can too)

- [x] Repository public for more than six months, with development spread
      over that period, releases, issues, and pull requests (on PyPI since
      March 2025, tags v0.1.1 to v1.0.0).
- [ ] **Research impact evidence**: papers or preprints that use the
      package, other groups that use it, or integration into research
      workflows. "Potential future use is not sufficient." See 3.6.
- [ ] Open-source practice signals: issues, pull requests, public
      discussion, **a CONTRIBUTING file, and stated support expectations**.
- [x] Iterative development, not a one-time code dump. JOSS runs automated
      checks on the commit distribution. The October 2026 work is
      concentrated, so the history before it matters ("OK: bursty
      activity" over 6+ months; "not acceptable: most commits in the last
      few weeks").

## 2. Repository

### 2.1 Basics

- [x] OSI-approved license as a plain-text file (LICENSE, MIT).
- [x] Public, cloneable without registration, open issue tracker, anyone
      can open issues and pull requests.
- [x] Packaged to community standards, `pip install`able (PyPI,
      `pyproject.toml`).
- [x] Installation instructions and dependencies (README,
      `pyproject.toml`).
- [x] Example usage, ideally solving a real analysis problem (README).
- [x] Automated tests with CI (GitHub Actions, Linux/macOS/Windows,
      Python 3.11 to 3.14).

### 2.2 Documentation

- [ ] Link the API documentation site from the README (no link yet).
- [ ] API documentation: "Good" means all functions documented with
      example inputs and outputs. Check that every public function has an
      Examples section.
- [ ] README states the statement of need (problem, audience, relation to
      other work), not only the paper.
- [ ] Reviewers must install the package and verify its core
      functionality, but most have no WRDS access. Document in the README
      how to run the full workflow with `"Pseudo Data"` and open data.

### 2.3 Community guidelines (three separate review items)

- [ ] `CONTRIBUTING.md`: how to contribute.
- [ ] How to report issues or problems (e.g., in `CONTRIBUTING.md` or
      issue templates).
- [ ] How to get support (e.g., GitHub Discussions or issues).
- [ ] Link all three from the README.
- [ ] `CODE_OF_CONDUCT.md` (expected good practice; reviewers also
      confirm the JOSS code of conduct).
- [ ] `CITATION.cff` (optional; add the JOSS DOI after acceptance).

### 2.4 Collaboration evidence (reviewers check the commit history)

- [ ] Submitting author made substantial contributions, as shown by the
      commit history. Decide who submits accordingly.
- [ ] Author list matches the contributions visible in the history; be
      ready to explain non-code contributions (project direction counts,
      funding or supervision alone does not).
- [ ] Code review through pull requests and responses to issues are
      visible (merge via PRs, not only direct pushes to `main`).

### 2.5 Paper branch

- [ ] JOSS requires a paper branch to be created from the default branch,
      so that it contains the source. Merge `main` into `joss-paper` before
      submitting.
- [x] Add the Open Journals draft-PDF GitHub Action
      (`.github/workflows/draft-pdf.yml`).
- [ ] Check that the first run compiles and read the PDF (Actions tab,
      artifact "paper").
- [ ] Move the cross-language check (`alignment_checks/`) into the
      repository, or describe it in the paper so that reviewers can verify
      the claim.

## 3. Paper (`joss/paper.md`, `joss/paper.bib`)

### 3.1 Metadata

- [x] Title as plain text, no backticks.
- [x] Authors with ORCIDs and affiliations (optional: `ror` IDs for the
      universities).
- [ ] All co-authors agree to be listed and to be accountable for all
      aspects of the work.
- [ ] Confirm the corresponding author (currently Christoph Scheuch).
- [ ] Set `date` to the submission date (format: `5 November 2026`).

### 3.2 Length and required sections

- [x] 750 to 1,750 words (currently about 1,570).
- [x] Summary for a diverse, non-specialist audience, without jargon.
- [x] "Statement of need" with exactly this heading (EditorialBot checks
      for it): problem, target audience, relation to other work.
- [x] State of the field with a "build vs. contribute" justification.
- [x] Software design: trade-offs, architecture, why it matters.
- [ ] Research impact statement with evidence (see 3.6).
- [ ] AI usage disclosure complete (see 3.7).
- [x] Acknowledgements section.
- [x] Ends with a "References" heading.

### 3.3 Claims to verify (reviewers confirm functional and performance claims)

- [ ] "334 datasets from 13 providers" (count with
      `list_supported_datasets()` on the submitted version).
- [x] Open Asset Pricing: "212 cross-sectional signals" and "written in
      Python since version 2.0" (openassetpricing.com: 212 predictors;
      signals translated to Python in the October 2025 release, v2.0.0).
- [x] Performance claim on polars: rephrased as a design rationale
      without a performance claim.
- [x] Default screen timing (`timing='formation'`, warning for
      `'every_period'`) matches the code.
- [ ] Code example runs on `"Pseudo Data"` with the submitted version.
- [ ] The paper reports no numerical results. If any are added, reviewers
      must be able to reproduce them, including access to the data, so
      WRDS-only results do not work (data sharing policy).

### 3.4 Content risks

- [x] The review criteria state that the paper "should not include
      software documentation such as API functionality". The API bullet
      list in Software design is rewritten as three design decisions.
- [ ] The code example is still API-like; keep it (reviewers value a
      real-world example) unless the editor asks to cut it.

### 3.5 References

- [x] Add numpy (`harris.2020`, Nature) and cite it where the paper
      names numpy.
- [x] DOIs added: `fama.2023` (SSRN 4629613), `walter.2024`
      (SSRN 4164117), polars (Zenodo concept DOI).
- [ ] `walter.2024`: SSRN 4164117 circulated as "Non-Standard Errors in
      Portfolio Sorts" (2022); confirm with Patrick Weiss that "Methodological
      Uncertainty in Portfolio Sorts" (2024) is the current title of the
      same paper.
- [x] All in-text citations exist in `paper.bib`; no unused or duplicate
      entries.
- [x] Full venue names, no abbreviations.
- [x] polars cited with version 2.1.0; OSAP code updated to v2.1.0.
- [ ] Versions for wrds, yfinance, linearmodels, and pandas-datareader
      (optional; URLs are given). Set them to the versions current at
      submission.
- [ ] Update `frey.2026` once the R Journal decides.
- [ ] Run `@editorialbot check references` after submission.
- [ ] Cite all prior similar work; reviewers point out missing tools.

### 3.6 Research impact statement

- [ ] Book claim: the book appeared in 2024, before the package was on
      PyPI, so cite tidy-finance.org for usage, not the book.
- [ ] Courses that use the package (university, course name, students).
- [ ] PyPI downloads with the period (e.g., pepy.tech).
- [ ] Papers, preprints, theses, or projects that use the package.
- [ ] Non-author issues or pull requests (strong positive signal).
- [ ] Factor library: confirm how it relates to the package and its size.

### 3.7 AI usage disclosure

- [ ] Tools and models with versions, and where they were used (code,
      tests, docs, paper text).
- [ ] Nature and scope of the assistance.
- [ ] How the quality and correctness of AI-assisted output was verified.
- [ ] Statement that the authors reviewed, edited, and validated all
      AI-assisted output and made the core design decisions.
- [ ] Every author confirms and adds other tools (e.g., GitHub Copilot).
      An incomplete disclosure counts as an ethical breach.

### 3.8 Acknowledgements

- [x] Danish Finance Institute (Stefan Voigt), with no role of the sponsor.
- [ ] Funding of the other authors, or a statement that there was none.

## 4. Submission form (https://joss.theoj.org/papers/new)

- [ ] Repository URL, branch `joss-paper`, software version.
- [ ] Disclose related publications: the R Journal submission on the R
      package (`frey.2026`), and explain how the JOSS paper differs.
- [ ] Conflicts of interest of all authors (financial, personal,
      professional), or a statement that there are none.
- [ ] Optional: preferred editor; suggested reviewers (do not @-tag them).

## 5. During review (public GitHub issue in openjournals/joss-reviews)

- [ ] Answer reviewer comments within 2 weeks; complete changes within
      4 to 6 weeks, or tell the editor.
- [ ] Reviewers file issues in this repository; fix them there and link
      back to the review issue. Acknowledge reviewer contributions.
- [ ] Regenerate the proof after changes with `@editorialbot generate pdf`
      (command on the first line of a comment).
- [ ] No AI for conversations with editors or reviewers (translation
      only).

## 6. After acceptance

- [ ] New tagged release if the software changed during review.
- [ ] Archive the release on Zenodo (or figshare) and get a DOI. The
      archive title and author list must match the paper; explain any
      difference.
- [ ] Post the version number and the archive DOI in the review issue.
- [ ] Proof-read the final proof (author list changes after publication
      are hard).
- [ ] Add the JOSS DOI to `CITATION.cff` and the README.

## Option to consider

pyOpenSci reviews Python packages; a package accepted by pyOpenSci gets a
streamlined JOSS review (the JOSS editor reuses the pyOpenSci review).
