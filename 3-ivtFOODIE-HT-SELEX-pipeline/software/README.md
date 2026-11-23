# External software

## HTS-IBIS

`HTS-IBIS/HTS-IBIS.jar` is installed from the authors' official executable
distribution:

- <https://acgt.cs.tau.ac.il/hts-ibis/>
- distribution: <https://acgt.cs.tau.ac.il/hts-ibis/HTS-IBIS.zip>

Reinstall reproducibly from the pipeline root:

```bash
mkdir -p software/HTS-IBIS
curl -fL https://acgt.cs.tau.ac.il/hts-ibis/HTS-IBIS.zip \
  -o software/HTS-IBIS/HTS-IBIS.zip
unzip -o software/HTS-IBIS/HTS-IBIS.zip -d software/HTS-IBIS
find software/HTS-IBIS -name 'HTS-IBIS.jar' -print
sha256sum software/HTS-IBIS/HTS-IBIS.jar
```

The downloaded archive is retained for provenance. Its SHA-256 checksum is
recorded in `software/HTS-IBIS/SHA256SUMS`.

## BEEML-SELEX / BEESEM status

No maintained Conda package or stable, versioned standalone distribution of
the original BEEML-SELEX was found. BEEML also assumes that the binding-site
position is known, which is inappropriate for an unknown motif within this
40-nt randomized region.

BEESEM is the published HT-SELEX successor that infers binding-site position,
but the available implementation (`https://github.com/sx-ruan/BEESEM`) depends
on Python 2.7 and the obsolete OpenOpt/FuncDesigner stack. Installing it into
the Python 3.11 production environment would make the environment fragile and
non-reproducible. It is therefore not installed or silently substituted.

The supported quantitative alternatives in this workflow are:

1. HTS-IBIS across cycle0-cycle4 for an HT-SELEX-specific PWM.
2. STREME with cycle4 as signal and cycle0 as matched control.
3. Raw and depth-normalized sequence-level enrichment tables for Seq2Kd or a
   modern custom energy-model implementation.

For a separate historical BEESEM environment, see the upstream repository and
use a sampled dataset; the method can require approximately 128 GB RAM for long
motifs and is not part of `snakemake --cores 32`.
