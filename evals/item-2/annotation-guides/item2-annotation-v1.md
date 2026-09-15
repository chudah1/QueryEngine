# Item 2 Claim Annotation Guide v1

## Purpose

Create a source-complete, consistent answer key for material claims in Part I,
Item 2 of a 10-Q. Annotators judge the filing text, not an extraction model.

## Include

Include claims about:

- company, segment, or product financial results;
- stated causes of changes in results;
- gross margin, operating expenses, and taxes;
- liquidity, contractual obligations, and capital returns;
- management expectations; and
- company-specific risks with a stated financial or operating impact.

## Exclude

Exclude:

- forward-looking-statement disclaimers;
- accounting-policy boilerplate without a current expected impact;
- website, filing, and procedural descriptions;
- generic industry background with no company impact; and
- duplicate claims that express the same fact for the same period.

## Atomicity

Create one claim for one financial fact, driver, outlook, or risk. Split text
when it contains facts with different types. For example, a current dividend
amount is `capital_allocation`, while an intention to increase a dividend is a
separate `capital_allocation` claim.

A result and its stated cause may remain one `business_driver` claim because the
cause is meaningful only in relation to that result.

## Claim types

- `financial_result`: A reported metric, balance, expense, tax position, or
  contractual obligation without a causal explanation.
- `business_driver`: A factor that caused or partly caused a reported change. A
  claim containing both a change and its stated cause uses this type.
- `management_outlook`: Management's expectation about future operating or
  financial performance or capacity.
- `risk`: A condition or event that may cause an adverse outcome.
- `capital_allocation`: A dividend, share repurchase, debt, or other decision
  about deploying or returning capital. Purchase obligations and tax payments
  are `financial_result`, not `capital_allocation`.

## Periods and values

- Preserve every material value and comparison period.
- Assert a period only when the evidence supports it.
- Do not merge quarterly and year-to-date periods when their drivers differ.
- Do not calculate values or infer comparisons not stated in the source.

## Evidence

- Copy one exact, continuous substring from the section.
- The quote must support the entire claim.
- Include the minimum context needed to resolve phrases such as “these trends,”
  “this change,” and “as a result.”
- Preserve punctuation, symbols, and characters such as `®` exactly.
- Prefer the shortest self-contained quote that uniquely supports the claim.

## Completeness pass

After claim-level review, scan every Item 2 subsection in order:

1. Business and macroeconomic conditions
2. Segment performance
3. Product and Services performance
4. Gross margins
5. Operating expenses
6. Income taxes
7. Liquidity and contractual obligations
8. Capital return program
9. Accounting pronouncements and estimates

Record supported omissions as missed claims before finalizing the gold case.
