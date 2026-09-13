# absfix2 stage-1 record

Written 2026-09-13T08:18:01+0200. Source: `runs/realdata/absfix2/stage1_record.json`.
Confirmatory scenes read from the `_seedbox` rerun (spec v2 13.15/13.16);
the calibration scene has plain entries only.

| scene | pfx | prefix job/state | T1 pooled window | T1 pre U/G | G members | S2 members / refusal | memb pre (P/R) | draws A/B | L refused (elig/truth mass) |
|---|---|---|---|---|---|---|---|---|---|
| flame_steak | s0 | 57485022 COMPLETED | [60,89] | True/True | 6582 | 6434 | True (1.0000/0.9775) | Y/Y | yes (2299/5668 = 0.41) |
| flame_steak | s1 | 57485047 COMPLETED | [60,90] | True/False | 6584 | 6450 | True (0.9998/0.9795) | Y/Y | yes (2221/5724 = 0.39) |
| flame_steak | s2 | 57485053 COMPLETED | [60,89] | True/True | 6648 | 6515 | True (1.0000/0.9800) | Y/Y | yes (2087/5393 = 0.39) |
| flame_steak | s3 | 57485054 COMPLETED | [60,93] | False/False | 6641 | 6497 | True (1.0000/0.9783) | Y/Y | yes (2112/5677 = 0.37) |
| sear_steak | s0 | 57504087 COMPLETED | [60,89] | -/- | 7676 | REFUSED (EXPECTED, 13.16) | - (-/-) | Y/Y | yes (2188/5706 = 0.38) |
| sear_steak | s1 | 57485062 COMPLETED | [60,89] | -/- | 7434 | REFUSED (EXPECTED, 13.16) | - (-/-) | Y/Y | yes (2259/5647 = 0.40) |
| sear_steak | s2 | 57504132 COMPLETED | [57,106] | -/- | 7656 | REFUSED (EXPECTED, 13.16) | - (-/-) | Y/Y | yes (2126/5782 = 0.37) |
| sear_steak | s3 | 57504146 COMPLETED | [60,90] | -/- | 7536 | REFUSED (EXPECTED, 13.16) | - (-/-) | Y/Y | yes (2289/5794 = 0.40) |
| cut_roasted_beef (calib) | s0 | - | - | True/False | - | 6927 | True (1.0000/0.9890) | - | - |
| cut_roasted_beef (calib) | s1 | - | - | True/False | - | 6917 | True (0.9996/0.9904) | - | - |
| cut_roasted_beef (calib) | s2 | - | - | True/False | - | 6778 | True (0.9999/0.9882) | - | - |
| cut_roasted_beef (calib) | s3 | - | - | True/False | - | 6751 | True (0.9996/0.9905) | - | - |

## Job states

| jobid | name | state | elapsed | exit |
|---|---|---|---|---|
| 57485022 | absfix2_flame_steak_uprefix6k_s0 | COMPLETED | 02:03:29 | 0:0 |
| 57485047 | absfix2_flame_steak_uprefix6k_s1 | COMPLETED | 02:11:26 | 0:0 |
| 57485053 | absfix2_flame_steak_uprefix6k_s2 | COMPLETED | 02:13:41 | 0:0 |
| 57485054 | absfix2_flame_steak_uprefix6k_s3 | COMPLETED | 02:01:59 | 0:0 |
| 57485060 | absfix2_sear_steak_uprefix6k_s0 | CANCELLED by 132193 | 03:46:21 | 0:0 |
| 57485062 | absfix2_sear_steak_uprefix6k_s1 | COMPLETED | 02:06:26 | 0:0 |
| 57485068 | absfix2_sear_steak_uprefix6k_s2 | CANCELLED by 132193 | 03:46:21 | 0:0 |
| 57485069 | absfix2_sear_steak_uprefix6k_s3 | CANCELLED by 132193 | 03:45:38 | 0:0 |
| 57485392 | absfix2_t1_flame_steak_s0 | CANCELLED by 132193 | 03:47:03 | 0:0 |
| 57485397 | absfix2_t1_flame_steak_s1 | COMPLETED | 01:50:57 | 0:0 |
| 57485398 | absfix2_vote_flame_steak_s1 | COMPLETED | 00:03:52 | 0:0 |
| 57485399 | absfix2_elig_flame_steak_s1 | COMPLETED | 00:01:15 | 0:0 |
| 57485401 | absfix2_t1_flame_steak_s2 | COMPLETED | 02:03:00 | 0:0 |
| 57485402 | absfix2_vote_flame_steak_s2 | CANCELLED by 132193 | 00:02:42 | 0:0 |
| 57485405 | absfix2_t1_flame_steak_s3 | COMPLETED | 01:55:25 | 0:0 |
| 57485406 | absfix2_vote_flame_steak_s3 | FAILED | 00:03:40 | 1:0 |
| 57485421 | absfix2_t1_sear_steak_s1 | COMPLETED | 01:54:03 | 0:0 |
| 57485422 | absfix2_vote_sear_steak_s1 | COMPLETED | 00:06:24 | 0:0 |
| 57485427 | absfix2_elig_sear_steak_s1 | COMPLETED | 00:02:51 | 0:0 |
| 57485503 | absfix2_smoke_vote | CANCELLED by 132193 | 00:28:28 | 0:0 |
| 57487886 | absfix2_smoke_vote | COMPLETED | 00:08:08 | 0:0 |
| 57487888 | absfix2_smoke_elig | COMPLETED | 00:02:48 | 0:0 |
| 57487916 | absfix2_smoke_draws | FAILED | 00:00:16 | 2:0 |
| 57488863 | absfix2_s2vote_cut_roasted_beef_s0 | COMPLETED | 00:06:10 | 0:0 |
| 57488864 | absfix2_s2vote_cut_roasted_beef_s1 | CANCELLED by 132193 | 00:38:06 | 0:0 |
| 57488865 | absfix2_s2vote_cut_roasted_beef_s2 | COMPLETED | 00:10:38 | 0:0 |
| 57488867 | absfix2_s2vote_cut_roasted_beef_s3 | CANCELLED by 132193 | 00:38:06 | 0:0 |
| 57488903 | absfix2_s2vote_flame_steak_s0 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57488906 | absfix2_s2vote_flame_steak_s1 | COMPLETED | 00:02:04 | 0:0 |
| 57488907 | absfix2_s2vote_flame_steak_s2 | CANCELLED by 132193 | 00:02:42 | 0:0 |
| 57488908 | absfix2_s2vote_flame_steak_s3 | FAILED | 00:04:32 | 1:0 |
| 57488909 | absfix2_s2vote_sear_steak_s0 | CANCELLED | 00:00:00 | 0:0 |
| 57488910 | absfix2_s2vote_sear_steak_s1 | FAILED | 00:05:52 | 1:0 |
| 57488911 | absfix2_s2vote_sear_steak_s2 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57488912 | absfix2_s2vote_sear_steak_s3 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57488990 | absfix2_precond_cut_roasted_beef_s0 | COMPLETED | 00:00:11 | 0:0 |
| 57489429 | absfix2_precond_cut_roasted_beef_s2 | COMPLETED | 00:00:11 | 0:0 |
| 57489499 | absfix2_precond_flame_steak_s1 | COMPLETED | 00:00:16 | 0:0 |
| 57489517 | absfix2_draws_flame_steak_s1 | COMPLETED | 00:01:19 | 0:0 |
| 57489525 | absfix2_draws_sear_steak_s1 | COMPLETED | 00:01:04 | 0:0 |
| 57492150 | absfix2_s2vote_cut_roasted_beef_s1 | COMPLETED | 00:04:54 | 0:0 |
| 57492151 | absfix2_precond_cut_roasted_beef_s1 | COMPLETED | 00:01:07 | 0:0 |
| 57492188 | absfix2_s2vote_cut_roasted_beef_s3 | CANCELLED by 132193 | 00:37:34 | 0:0 |
| 57494941 | absfix2_s2vote_cut_roasted_beef_s3 | TIMEOUT | 01:33:20 | 0:0 |
| 57504087 | absfix2_sear_steak_uprefix6k_s0 | COMPLETED | 02:02:00 | 0:0 |
| 57504113 | absfix2_t1_sear_steak_s0 | COMPLETED | 01:56:12 | 0:0 |
| 57504114 | absfix2_vote_sear_steak_s0 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504119 | absfix2_elig_sear_steak_s0 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504120 | absfix2_draws_sear_steak_s0 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504125 | absfix2_s2vote_sear_steak_s0 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504126 | absfix2_precond_sear_steak_s0 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504132 | absfix2_sear_steak_uprefix6k_s2 | COMPLETED | 02:02:44 | 0:0 |
| 57504133 | absfix2_t1_sear_steak_s2 | COMPLETED | 02:03:01 | 0:0 |
| 57504134 | absfix2_vote_sear_steak_s2 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504136 | absfix2_elig_sear_steak_s2 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504137 | absfix2_draws_sear_steak_s2 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504138 | absfix2_s2vote_sear_steak_s2 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504144 | absfix2_precond_sear_steak_s2 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504146 | absfix2_sear_steak_uprefix6k_s3 | COMPLETED | 02:11:20 | 0:0 |
| 57504147 | absfix2_t1_sear_steak_s3 | COMPLETED | 01:57:08 | 0:0 |
| 57504154 | absfix2_vote_sear_steak_s3 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504155 | absfix2_elig_sear_steak_s3 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504157 | absfix2_draws_sear_steak_s3 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504163 | absfix2_s2vote_sear_steak_s3 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504169 | absfix2_precond_sear_steak_s3 | CANCELLED by 132193 | 00:00:00 | 0:0 |
| 57504222 | absfix2_s2vote_cut_roasted_beef_s3 | COMPLETED | 00:04:28 | 0:0 |
| 57504225 | absfix2_precond_cut_roasted_beef_s3 | COMPLETED | 00:01:00 | 0:0 |
| 57508186 | absfix2_seedbox | COMPLETED | 00:07:00 | 0:0 |
| 57508564 | absfix2_seedbox2 | COMPLETED | 00:07:35 | 0:0 |
| 57509030 | absfix2_vote_flame_steak_s0_seedbox | CANCELLED by 132193 | 01:11:39 | 0:0 |
| 57509031 | absfix2_elig_flame_steak_s0_seedbox | CANCELLED | 00:00:00 | 0:0 |
| 57509033 | absfix2_draws_flame_steak_s0_seedbox | CANCELLED | 00:00:00 | 0:0 |
| 57509034 | absfix2_s2vote_flame_steak_s0_seedbox | COMPLETED | 00:04:06 | 0:0 |
| 57509061 | absfix2_precond_flame_steak_s0_seedbox | CANCELLED | 00:00:00 | 0:0 |
| 57509062 | absfix2_vote_flame_steak_s1_seedbox | COMPLETED | 00:06:25 | 0:0 |
| 57509063 | absfix2_elig_flame_steak_s1_seedbox | COMPLETED | 00:01:08 | 0:0 |
| 57509064 | absfix2_draws_flame_steak_s1_seedbox | COMPLETED | 00:00:12 | 0:0 |
| 57509065 | absfix2_s2vote_flame_steak_s1_seedbox | CANCELLED by 132193 | 01:11:39 | 0:0 |
| 57509067 | absfix2_precond_flame_steak_s1_seedbox | CANCELLED | 00:00:00 | 0:0 |
| 57509068 | absfix2_vote_flame_steak_s2_seedbox | COMPLETED | 00:06:56 | 0:0 |
| 57509069 | absfix2_elig_flame_steak_s2_seedbox | COMPLETED | 00:02:53 | 0:0 |
| 57509070 | absfix2_draws_flame_steak_s2_seedbox | COMPLETED | 00:00:11 | 0:0 |
| 57509071 | absfix2_s2vote_flame_steak_s2_seedbox | COMPLETED | 00:04:06 | 0:0 |
| 57509072 | absfix2_precond_flame_steak_s2_seedbox | COMPLETED | 00:00:08 | 0:0 |
| 57509073 | absfix2_vote_flame_steak_s3_seedbox | COMPLETED | 00:08:12 | 0:0 |
| 57509075 | absfix2_elig_flame_steak_s3_seedbox | COMPLETED | 00:01:31 | 0:0 |
| 57509076 | absfix2_draws_flame_steak_s3_seedbox | COMPLETED | 00:00:12 | 0:0 |
| 57509077 | absfix2_s2vote_flame_steak_s3_seedbox | COMPLETED | 00:04:15 | 0:0 |
| 57509078 | absfix2_precond_flame_steak_s3_seedbox | COMPLETED | 00:00:11 | 0:0 |
| 57509079 | absfix2_vote_sear_steak_s0_seedbox | COMPLETED | 00:04:27 | 0:0 |
| 57509080 | absfix2_elig_sear_steak_s0_seedbox | COMPLETED | 00:02:46 | 0:0 |
| 57509081 | absfix2_draws_sear_steak_s0_seedbox | COMPLETED | 00:00:29 | 0:0 |
| 57509082 | absfix2_s2vote_sear_steak_s0_seedbox | FAILED | 00:03:38 | 1:0 |
| 57509083 | absfix2_precond_sear_steak_s0_seedbox | CANCELLED | 00:00:00 | 0:0 |
| 57509084 | absfix2_vote_sear_steak_s1_seedbox | COMPLETED | 00:04:01 | 0:0 |
| 57509085 | absfix2_elig_sear_steak_s1_seedbox | COMPLETED | 00:02:50 | 0:0 |
| 57509086 | absfix2_draws_sear_steak_s1_seedbox | COMPLETED | 00:00:12 | 0:0 |
| 57509087 | absfix2_s2vote_sear_steak_s1_seedbox | FAILED | 00:03:40 | 1:0 |
| 57509088 | absfix2_precond_sear_steak_s1_seedbox | CANCELLED | 00:00:00 | 0:0 |
| 57509090 | absfix2_vote_sear_steak_s2_seedbox | COMPLETED | 00:04:44 | 0:0 |
| 57509091 | absfix2_elig_sear_steak_s2_seedbox | COMPLETED | 00:02:44 | 0:0 |
| 57509092 | absfix2_draws_sear_steak_s2_seedbox | COMPLETED | 00:00:47 | 0:0 |
| 57509094 | absfix2_s2vote_sear_steak_s2_seedbox | FAILED | 00:04:03 | 1:0 |
| 57509098 | absfix2_precond_sear_steak_s2_seedbox | CANCELLED | 00:00:00 | 0:0 |
| 57509099 | absfix2_vote_sear_steak_s3_seedbox | COMPLETED | 00:03:59 | 0:0 |
| 57509100 | absfix2_elig_sear_steak_s3_seedbox | COMPLETED | 00:01:07 | 0:0 |
| 57509101 | absfix2_draws_sear_steak_s3_seedbox | COMPLETED | 00:00:11 | 0:0 |
| 57509102 | absfix2_s2vote_sear_steak_s3_seedbox | FAILED | 00:01:40 | 1:0 |
| 57509103 | absfix2_precond_sear_steak_s3_seedbox | CANCELLED | 00:00:00 | 0:0 |
| 57514412 | absfix2_t1_flame_steak_s0 | CANCELLED by 132193 | 00:12:32 | 0:0 |
| 57514413 | absfix2_vote_flame_steak_s0_seedbox | COMPLETED | 00:05:24 | 0:0 |
| 57514414 | absfix2_elig_flame_steak_s0_seedbox | COMPLETED | 00:02:51 | 0:0 |
| 57514415 | absfix2_draws_flame_steak_s0_seedbox | COMPLETED | 00:00:10 | 0:0 |
| 57514416 | absfix2_precond_flame_steak_s0_seedbox | CANCELLED | 00:00:00 | 0:0 |
| 57514417 | absfix2_s2vote_flame_steak_s1_seedbox | COMPLETED | 00:04:29 | 0:0 |
| 57514418 | absfix2_precond_flame_steak_s1_seedbox | COMPLETED | 00:01:57 | 0:0 |
| 57515412 | absfix2_t1_flame_steak_s0 | COMPLETED | 01:51:28 | 0:0 |
| 57515413 | absfix2_precond_flame_steak_s0_seedbox | COMPLETED | 00:00:58 | 0:0 |
