#!/usr/bin/env bash
# Pre-submission checks for an ARR long paper (run from anywhere).
# Covers the ARR desk-reject triggers in survey/T1_work/arr_rules.md.
# Non-zero exit = do not submit. Run after `tools/build.sh --final`.
# Not covered here (do by hand): citation audit (/citation-audit), aclpubcheck,
# Responsible NLP checklist, OpenReview form fields, supplementary archives.
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
say() { printf '%-62s %s\n' "$1" "$2"; }
check() { # name, command that prints offending lines (empty = pass)
  local out; out=$(eval "$2" 2>/dev/null)
  if [[ -n "$out" ]]; then say "$1" "FAIL"; echo "$out" | head -15 | sed 's/^/    /'; fail=1; else say "$1" "ok"; fi
}
warn() { say "$1" "WARN"; [[ -n "${2:-}" ]] && echo "$2" | sed 's/^/    /'; }
nocomment() { grep -vE '^[^:]+:[0-9]+:[[:space:]]*%'; }   # drop hits on comment lines
TEX="main.tex macros.tex sections/*.tex tables/*.tex"
BODY="sections/*.tex tables/*.tex"

echo "== draft leftovers"
check "draft switch is off (\\draftfalse active)"            "grep -L '^\\\\draftfalse' main.tex"
check "no \\tbd table cells"                                 "grep -Hn '\\\\tbd' $BODY | nocomment"
check "no \\placeholder / \\todo / \\note / \\unverified"     "grep -HnE '\\\\(placeholder|todo|note|unverified)\\{' $BODY | nocomment"
check "no \\figplaceholder"                                  "grep -Hn '\\\\figplaceholder' $BODY | nocomment"
check "no UNVERIFIED / TODO / XXX / ?? in text"              "grep -HnE 'UNVERIFIED|TODO|XXX|\\?\\?' $BODY | nocomment"
check "no UNVERIFIED bib entries"                            "grep -Hn 'UNVERIFIED' custom.bib | nocomment"
check "demo number key removed"                              "grep -Hn 'demo\\.' numbers/*.tex $BODY | nocomment"

echo "== format and mandatory sections"
check "review mode (\\usepackage[review]{acl})"             "grep -L 'usepackage\\[review\\]{acl}' main.tex"
check "style files unmodified (acl.sty / acl_natbib.bst)"    "git diff --stat HEAD -- acl.sty acl_natbib.bst; git log --format=%h -- acl.sty acl_natbib.bst | sed -n '2,\$p'"
check "section titled exactly 'Limitations' exists"          "grep -L '\\\\section\\*{Limitations}' sections/8_limitations.tex"
check "no figures/tables/results in Limitations"             "grep -HnE '\\\\begin\\{(figure|table)|\\\\res\\{|\\\\includegraphics' sections/8_limitations.tex sections/9_ethics.tex | nocomment"
check "no Acknowledgments section (review version)"          "grep -HniE '\\\\section\\*?\\{acknowledg' $TEX | nocomment"
check "appendix stays two-column (no \\onecolumn)"           "grep -HnE '\\\\onecolumn|\\\\begin\\{strip\\}' $TEX | nocomment"
check "Related Work in the main text"                        "grep -L '\\\\section{Related Work}' sections/2_related.tex"
check "no spacing hacks (neg. vspace, float/line spacing)"   "grep -HnE '\\\\vspace\\*?\\{ *-|\\\\vskip *-|\\\\(baselinestretch|linespread|textfloatsep|floatsep|intextsep|abovecaptionskip|belowcaptionskip|abovedisplayskip|belowdisplayskip|parskip|titlespacing|enlargethispage)|\\\\setlength\\{\\\\(textheight|textwidth|columnsep|topmargin|oddsidemargin)|\\\\setlist|\\\\captionsetup|geometry\\}' $TEX | nocomment"
check "no tiny fonts in body text (\\tiny, \\fontsize < 8)"   "grep -HnE '\\\\tiny|\\\\fontsize\\{[1-7](\\.[0-9])?pt' $TEX | nocomment"

echo "== anonymity and integrity"
check "anonymous author block"                               "grep -Hn '\\\\author{' main.tex | grep -v 'Anonymous'"
check "no self-identifying phrases (outside comments)"       "grep -HniE 'our (previous|prior|earlier) (work|paper)|we previously (showed|proposed)|github\\.com/[A-Za-z]|huggingface\\.co/[A-Za-z]|wandb\\.ai' $TEX | nocomment | grep -v 'anonymous'"
check "no cloud-drive links (use anonymous.4open.science)"   "grep -HniE 'drive\\.google|dropbox|onedrive|box\\.com|baidu\\.com/s|pan\\.baidu' $TEX custom.bib | nocomment"
check "no hidden / white / invisible text"                   "grep -HnE 'color\\{white\\}|textcolor\\{white\\}|\\\\phantom\\{[^}]{12,}|opacity|\\\\pdfliteral|transparent' $TEX | nocomment"
check "no prompt-injection strings"                          "grep -HniE 'ignore (all |the )?(previous|prior|above) instructions|as an? (ai|llm|language model) review|give (this|the) paper (a )?(high|positive)|reviewer.{0,20}(llm|gpt|chatgpt)' $TEX | nocomment"
check "no AppleDouble files tracked"                         "git ls-files | grep -E '(^|/)\\._'"

echo "== compiled PDF (build/)"
PDF=$(ls -t build/main_final.pdf build/main.pdf 2>/dev/null | head -1)
if [[ -n "$PDF" ]]; then
  LOG="${PDF%.pdf}.log"; [[ -f "$LOG" ]] || LOG="build/_final_main.log"
  [[ -f "$LOG" ]] && check "no undefined references/citations" "grep -E 'Reference .* undefined|Citation .* undefined|There were undefined' '$LOG'"
  check "PDF page size is A4" "pdfinfo '$PDF' | grep 'Page size' | grep -v '595.2[0-9]* x 841.8[0-9]*'"
  if command -v pdffonts >/dev/null; then
    check "all fonts embedded, no Type 3" "pdffonts '$PDF' | awk 'NR>2 && (\$(NF-4)!=\"yes\" || \$2==\"Type\" && \$3==\"3\")'"
  fi
  check "PDF metadata has no author name" "pdfinfo '$PDF' | grep -iE '^Author:.*[A-Za-z]' | grep -vi anonymous"
  # Limitations must start right after the 8 content pages.
  n=$(pdfinfo "$PDF" | awk '/^Pages:/{print $2}'); lp=""
  for ((p=1; p<=n; p++)); do
    if pdftotext -f $p -l $p "$PDF" - 2>/dev/null | grep -qx 'Limitations'; then lp=$p; break; fi
  done
  if [[ -z "$lp" ]]; then warn "Limitations heading located in PDF" "not found by pdftotext; check by eye"
  elif (( lp > 9 )); then say "content ends by page 8 (Limitations starts p$lp)" "FAIL"; fail=1
  else say "content ends by page 8 (Limitations starts p$lp)" "ok"; fi
  words=$(pdftotext -f 1 -l 1 "$PDF" - | awk '/^Abstract$/{f=1;next} /^1[[:space:]]+Introduction/{f=0} f' | tr -s '[:space:]' '\n' | grep -vcE '^[0-9]*$')
  (( words > 200 )) && { say "abstract <= 200 words (~$words)" "WARN"; } || say "abstract <= 200 words (~$words)" "ok"
  echo "   note: tectonic (XeTeX) and Overleaf (pdfLaTeX) can differ by a few lines; confirm the page count on Overleaf."
else
  warn "compiled PDF" "none in build/: run tools/build.sh --final first"
fi
echo "== manual (not automatable here)"
echo "   [ ] citation audit done (/citation-audit): every entry exists and supports its claim"
echo "   [ ] aclpubcheck passed: https://github.com/acl-org/aclpubcheck"
echo "   [ ] Responsible NLP checklist complete (checklist/responsible_nlp_checklist.md), E1 accurate"
echo "   [ ] supplementary: software and data as separate .zip/.tgz <= 200 MB each, anonymized"
exit $fail
