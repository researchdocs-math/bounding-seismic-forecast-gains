# Data provenance

The experiment preserves separate regional catalog cohorts. Time is UTC, depth is in kilometers, and magnitude is the provider's reported preferred value. Raw responses and query manifests preserve the acquisition snapshots from October 2026.

| Cohort | Provider and service | Processing |
|---|---|---|
| Italy | INGV ISIDe, https://webservices.ingv.it/fdsnws/event/1/ ; dataset DOI 10.13127/ISIDE | Annual preferred earthquake responses; 2000–2025; reported magnitude at least 3; depth 0–60 km. |
| Greece | National Observatory of Athens, https://www.gein.noa.gr/services/full_catalogue.php | Full text snapshot, filtered to 2000–2025, reported magnitude at least 3, depth 0–60 km. |
| Alaska | USGS ComCat, https://earthquake.usgs.gov/fdsnws/event/1/ | Annual earthquake responses; reported magnitude at least 3; declared geographic domain and depth 0–60 km. |
| Chile region | USGS ComCat, same service | Annual earthquake responses; reported magnitude at least 4.5; geographic domain and depth 0–60 km. |
| Japan region | USGS ComCat, same service | Annual earthquake responses; reported magnitude at least 4.5; geographic domain and depth 0–60 km. |

Italian and Greek snapshots and initial forecast matrices were retained from prior development work. All new feature experiments and all Alaska, Chile, and Japan cohorts were constructed for this study. The full reproduction mode rebuilds the matrices directly from the retained source responses. Catalog identifiers are deduplicated. Greek identifiers are digests of source records. The regional catalogs remain separate; no cross-provider magnitude conversion or event matching is applied.

For Chile and Japan, `reported_magnitude` retains the original value and `magnitude` subtracts 1.5 only to reuse the common internal ETAS threshold of 3.0. This is a numerical translation, not a conversion to moment magnitude. Target keys `ge35`, `band35_45`, and `ge45` correspond to reported M >= 5.0, 5.0 <= M < 6.0, and M >= 6.0 in those two regions. Elsewhere their reported thresholds are 3.5 and 4.5.

The MIT license covers the research software. Catalog records retain their providers' source terms and attribution requirements. INGV ISIDe data are identified by their dataset DOI and provider documentation; NOA records are attributed to the National Observatory of Athens; USGS source information is generally in the public domain, subject to any third-party notices in the source. Consult each provider's terms for uses beyond reproduction. The archive does not relicense third-party observations.
