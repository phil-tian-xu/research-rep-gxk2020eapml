
|Field|Meaning|
|---|---|
|`permno`|Permanent **security** identifier|
|`permco`|Permanent **company** identifier; multiple securities can share one|
|`date`|CRSP monthly observation date|
|`ret`|Total holding-period return including distributions, in decimals|
|`retx`|Return excluding dividends, in decimals|
|`prc`|Unadjusted month-end price in dollars; a negative sign indicates a bid/ask average|
|`shrout`|Shares outstanding, in **thousands**|
|`vol`|Legacy monthly trading volume, in **hundreds of shares**|
|`cfacpr`|Cumulative price-adjustment factor: adjusted price = `abs(prc) / cfacpr`|
|`cfacshr`|Cumulative share-adjustment factor: adjusted shares = `shrout × cfacshr`|
|`bidlo`|Lowest daily price or bid/ask average during the month|
|`askhi`|Highest daily price or bid/ask average during the month|
