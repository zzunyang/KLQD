const sleep = (second) =>
  new Promise((resolve) => setTimeout(resolve, second * 1000));
const puppeteer = require("puppeteer");
const fs = require("fs");
const fileName = "klac_CyberConsolution_재외.json";
const crawlklac = async () => {
  const browser = await puppeteer.launch({
    headless: false,
    args: ["--window-size=1920,1080"],
  });
  const page = await browser.newPage();

  try {
    const questionsAndAnswers = [];
    const url = `https://www.klac.or.kr/legalstruct/cyberConsultation/selectOpenArticleDetail.do?boardCode=2&contentId=US_0000110532&pageIndex=1&searchCnd=0&searchWrd=`;
    await page.goto(url);
    for (let currentPage = 1; currentPage <= 49; currentPage++) {
      await page.waitForSelector(".notice_contents"); // Wait for the question and answer elements to be available

      const data = await page.evaluate(() => {
        const title =
          document.querySelector(".view_head")?.textContent?.trim() || "";
        const question =
          document
            .querySelector(
              "#content > form:nth-child(1) > div.contents_doc > div > div:nth-child(8) > div.notice_contents"
            )
            ?.textContent?.trim() || "";
        const answer =
          document
            .querySelector(".notice_contents:nth-of-type(2)")
            ?.textContent?.trim() || "";
        const url = window.location.href;

        return { title, question, answer, url };
      });

      questionsAndAnswers.push(data);
      await sleep(1);

      console.log(`현재 페이지: ${currentPage}`);
      await page.click("div.page_foot>dl:nth-child(2)>dd>span.pf_tit>a"); // Change this selector to the appropriate next page selector
    }

    const jsonContent = JSON.stringify(questionsAndAnswers, null, 2);
    fs.writeFileSync(fileName, jsonContent, "utf-8");

    console.log(
      `대한법률구조공단 사이버상담 페이지 데이터를 성공적으로 크롤링하여 ${fileName} 파일로 저장했습니다.`
    );
  } catch (error) {
    console.error("크롤링 도중 오류가 발생했습니다:", error);
  } finally {
    await browser.close();
  }
  // }
};
crawlklac();
