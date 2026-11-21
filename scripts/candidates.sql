set statement_timeout='1800s'; set work_mem='96MB';
create temp table f as
select product_id, review_id, content, rating, author, written_at::date wd, option, cardinality(images) imgs, helpful_count,
 char_length(regexp_replace(content,'[^0-9A-Za-z가-힣]','','g')) mlen,
 md5(regexp_replace(lower(content),'[^0-9a-z가-힣]','','g')) h
from reviews where platform='__PF__';
create temp table dup as select h, count(distinct review_id) ids, count(distinct product_id) prods from f where mlen>=30 group by h having count(distinct review_id)>=2 and count(distinct product_id)>=2;
create temp table dd as select product_id, wd, count(*) c from f group by 1,2;
create temp table pm as select product_id, avg(c) mean, sum(c) total from dd group by 1;
create temp table au as select author, count(*) n, count(distinct product_id) prods from f where author is not null and author<>'' and author not like '%*%' group by 1;
\copy (select '__PF__' platform, f.product_id, review_id, content, rating, author, wd written_date, option, imgs, helpful_count, mlen, dd.c day_c, round(pm.mean,2) day_mean, pm.total prod_total, coalesce(dup.ids,0) dup_ids, coalesce(dup.prods,0) dup_prods, au.n author_n, au.prods author_prods from f left join dup using(h) left join dd using(product_id, wd) left join pm using(product_id) left join au using(author)) to stdout with csv header
