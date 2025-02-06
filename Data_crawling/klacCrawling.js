const puppeteer = require("puppeteer");
const fs = require("fs");
const createCsvWriter = require("csv-writer").createObjectCsvWriter;
let i = 1;
let j = 16; //https://www.klac.or.kr/legalinfo/counsel.do
let limit = 183;
let fileName = "법률상담사례-채권.json";
const csvWriter = createCsvWriter({
  path: `법률상담사례-보전처분.csv`,
  header: [
    { id: "instruction", title: "Title" },
    { id: "output", title: "Content" },
  ],
  append: true,
});
const getArticles = async () => {
  const browser = await puppeteer.launch({
    headless: false,
  });
  const page = await browser.newPage();
  let articles = [];

  for (i; i <= limit; i++) {
    if (i < 10) {
      await page.goto(
        `https://www.klac.or.kr/legalinfo/counselView.do?folderId=001&scdFolderId=&pageIndex=1&searchCnd=&searchWrd=&caseId=case-0${j}-0000${i}`
      );
    } else if (i < 100) {
      await page.goto(
        `https://www.klac.or.kr/legalinfo/counselView.do?folderId=001&scdFolderId=&pageIndex=1&searchCnd=&searchWrd=&caseId=case-0${j}-000${i}`
      );
    } else if (i < 1000) {
      await page.goto(
        `https://www.klac.or.kr/legalinfo/counselView.do?folderId=001&scdFolderId=&pageIndex=1&searchCnd=&searchWrd=&caseId=case-0${j}-00${i}`
      );
    } else {
      await page.goto(
        `https://www.klac.or.kr/legalinfo/counselView.do?folderId=001&scdFolderId=&pageIndex=1&searchCnd=&searchWrd=&caseId=case-0${j}-0${i}`
      );
    }
    await page.waitForSelector("#print_page > div:nth-child(3) > dl > dd");

    const article = await page.evaluate(() => {
      const titleElements = document.querySelectorAll(
        "#print_page > div:nth-child(3) > dl > dd"
      );
      const contentElements = document.querySelectorAll(
        "#print_page > div:nth-child(4) > dl > dd"
      );

      const titleArr = Array.from(titleElements, (title) => title.textContent);
      const contentArr = Array.from(contentElements, (content) =>
        content.textContent
          .replace(/\t/g, "")
          .replace(/\n/g, "")
          .replace(
            "※ 주의 : 사례에 대한 답변은 법령이나 판례 등의 변경으로 내용이 바뀔 수 있으므로 구체적인 사안에 대해서는 반드시 대한법률구조공단 상담(전화상담은 국번없이 ☎ 132) 등을 통해 다시 한 번 확인하시기 바랍니다.",
            ""
          )
      );

      return { titleArr, contentArr };
    });

    articles.push(article);
  }

  await browser.close();
  return articles;
};

(async () => {
  const articles = await getArticles();

  if (articles.length === 0) {
    console.log("No articles found.");
    return;
  }

  const records = [];

  articles.forEach((article) => {
    const { titleArr, contentArr } = article;
    for (let i = 0; i < titleArr.length; i++) {
      records.push({
        instruction: titleArr[i],
        input: "",
        output: contentArr[i],
      });
    }
  });
  fs.writeFileSync(fileName, JSON.stringify(records, null, 2));
  // csvWriter.writeRecords(records).then(() => {
  //   // console.log(JSON.stringify(records));
  // });
  console.log("Json 파일이 정상적으로 작성되었습니다.");
})();

// 위 코드를 현재 페이지에 ul.question > li.qa가 존재하지 않을 시 종료하도록 수정
