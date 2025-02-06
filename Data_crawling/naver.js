// var express = require("express");
// var fs = require("fs");
// var app = express();

// const sleep = (second) =>
//   new Promise((resolve) => setTimeout(resolve, second * 1000));
// const puppeteer = require("puppeteer");
// const fs = require("fs");
// const fileName = "naver_kin_data_산업재해_231122.json";
// var client_id = "_RNhsfOvwtZEFkbOGadb";
// var client_secret = "N8gJ2l0R5O";
// const naver = () => {
//   let fileName = "naver_kin_search_이혼_1.json";
//   var api_url = "https://openapi.naver.com/v1/search/kin.json";
//   var request = require("request");
//   var options = {
//     url: api_url,
//     headers: {
//       "X-Naver-Client-Id": client_id,
//       "X-Naver-Client-Secret": client_secret,
//     },
//     method: "GET",
//     encoding: "utf-8",
//     url: api_url,
//     qs: { query: "이혼", display: 100, start: 100 },
//   };
//   // request.get(options, function (error, response, body) {
//   //   if (!error && response.statusCode == 200) {
//   //     res.writeHead(200, { "Content-kinURL": "text/json;charset=utf-8" });
//   //     res.end(body);
//   //   } else {
//   //     res.status(response.statusCode).end();
//   //     console.log("error = " + response.statusCode);
//   //   }
//   // });
//   request(options, async function (error, response, body) {
//     body = body.replace(/<b>/gi, "");
//     body = body.replace(/<\/b>/, "");
//     // fs.writeFileSync(fileName, body.toString(), "utf-8");
//     // console.log(body);
//     const browser = await puppeteer.launch({
//       headless: false,
//       args: ["--window-size=1920,1080"],
//     });
//     const page = await browser.newPage();

//     try {
//       const questionsAndAnswers = [];
//       // for (let currentPage = 99; currentPage >= 1; currentPage--) {
//       //   const url = `https://kin.naver.com/qna/expertAnswerList.naver?dirId=60225&queryTime=2023-11-21%2013%3A51%3A23&page=${currentPage}`;
//       //   await page.goto(url);

//       const questionLinks = JSON.parse(body)["items"].map(
//         (item) => item["link"]
//       );

//       for (const link of questionLinks) {
//         await page.goto(link);

//         const data = await page.evaluate(() => {
//           const title =
//             document
//               .querySelector(
//                 "#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div"
//               )
//               ?.textContent?.trim() || "";
//           const question =
//             document
//               .querySelector(
//                 "#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content"
//               )
//               ?.textContent?.trim() || "";
//           const answer =
//             document
//               .querySelector(
//                 "#answer_1 > div._endContents.c-heading-answer__content > div._endContentsText.c-heading-answer__content-user > div > div"
//               )
//               ?.textContent?.trim() || "";
//           const url = window.location.href;

//           const instruction = `${title} ${question}`;
//           const output = answer;
//           const kinURL = url;
//           return { instruction, output, kinURL };
//         });

//         questionsAndAnswers.push(data);
//         sleep(0.5);
//       }
//       // console.log(`현재 페이지: ${currentPage}`);
//       // }

//       const jsonContent = JSON.stringify(questionsAndAnswers, null, 2);
//       fs.writeFileSync(fileName, jsonContent, "utf-8");

//       console.log(
//         `네이버 지식인 페이지 데이터를 성공적으로 크롤링하여 ${fileName} 파일로 저장했습니다.`
//       );
//     } catch (error) {
//       console.error("크롤링 도중 오류가 발생했습니다:", error);
//     } finally {
//       await browser.close();
//     }
//     // questionsLink = JSON.parse(body)["link"];
//   });
// };
// naver();
// 특정키워드가 있는곳만 출력되서 폐기
const sleep = (second) =>
  new Promise((resolve) => setTimeout(resolve, second * 1000));
const puppeteer = require("puppeteer");
const fs = require("fs");
const fileName = "naver_kin_data_세금&세무_연말정산_231122.json";
const crawlNaverKin = async () => {
  const browser = await puppeteer.launch({
    headless: false,
    args: ["--window-size=1920,1080"],
  });
  const page = await browser.newPage();

  try {
    const questionsAndAnswers = [];
    for (let currentPage = 99; currentPage >= 1; currentPage--) {
      const url = `https://kin.naver.com/qna/expertAnswerList.naver?dirId=40311&queryTime=2023-11-21%2013%3A51%3A23&page=${currentPage}`;
      await page.goto(url);

      const questionLinks = await page.$$eval(
        "#au_board_list > tr > td.title > a",
        (links) =>
          links.map((link) => {
            const relativeURL = link.getAttribute("href");
            const absoluteURL = new URL(relativeURL, window.location.href).href;
            return absoluteURL;
          })
      );

      for (const link of questionLinks) {
        await page.goto(link);

        const data = await page.evaluate(() => {
          const title =
            document
              .querySelector(
                "#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div"
              )
              ?.textContent?.trim() || "";
          const question =
            document
              .querySelector(
                "#content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__content"
              )
              ?.textContent?.trim() || "";
          const answer =
            document
              .querySelector(
                "#answer_1 > div._endContents.c-heading-answer__content > div._endContentsText.c-heading-answer__content-user > div > div"
              )
              ?.textContent?.trim() || "";
          const url = window.location.href;

          const instruction = `${title} ${question}`;
          const output = answer;
          const kinURL = url;
          return { instruction, output, kinURL };
        });

        questionsAndAnswers.push(data);
        sleep(0.5);
      }
      console.log(`현재 페이지: ${currentPage}`);
    }

    const jsonContent = JSON.stringify(questionsAndAnswers, null, 2);
    fs.writeFileSync(fileName, jsonContent, "utf-8");

    console.log(
      `네이버 지식인 페이지 데이터를 성공적으로 크롤링하여 ${fileName} 파일로 저장했습니다.`
    );
  } catch (error) {
    console.error("크롤링 도중 오류가 발생했습니다:", error);
  } finally {
    await browser.close();
  }
  // }
};
crawlNaverKin();
